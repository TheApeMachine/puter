//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalPageWrite(
	storage tensor.Tensor,
	values tensor.Tensor,
	pageIDs tensor.Tensor,
	offsets tensor.Tensor,
	pageSize tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPageWrite(storage, values, pageIDs, offsets, pageSize, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second, config.third)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_page_write(
		config.first.bridge.device,
		C.int(config.elementDType),
		config.first.buffer,
		config.second.buffer,
		config.third.buffer,
		config.offsets.buffer,
		config.out.buffer,
		C.uint32_t(config.pageCount),
		C.uint32_t(config.pageSize),
		C.uint32_t(config.inner),
		C.uint32_t(config.rows),
		C.uint64_t(token),
		&status,
	)

	return finishMetalShapeIndexDispatch("page_write", token, rc, status)
}

func runMetalPageGather(
	storage tensor.Tensor,
	pageTable tensor.Tensor,
	pageSize tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPageGather(storage, pageTable, pageSize, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_page_gather(
		config.first.bridge.device,
		C.int(config.elementDType),
		config.first.buffer,
		config.second.buffer,
		config.out.buffer,
		C.uint32_t(config.pageCount),
		C.uint32_t(config.pageSize),
		C.uint32_t(config.inner),
		C.uint32_t(config.rows),
		C.uint64_t(token),
		&status,
	)

	return finishMetalShapeIndexDispatch("page_gather", token, rc, status)
}

type metalPageConfig struct {
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	offsets      *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	pageCount    uint32
	pageSize     uint32
	inner        uint32
	rows         uint32
}

func requireMetalPageWrite(
	storage tensor.Tensor,
	values tensor.Tensor,
	pageIDs tensor.Tensor,
	offsets tensor.Tensor,
	pageSize tensor.Tensor,
	out tensor.Tensor,
) (metalPageConfig, error) {
	storageTensor, pageIDTensor, outTensor, err := requireMetalShapeIndexBase(storage, pageIDs, out)
	if err != nil {
		return metalPageConfig{}, err
	}

	valueTensor, ok := values.(*metalTensor)
	if !ok {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	offsetTensor, ok := offsets.(*metalTensor)
	if !ok {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	pageSizeValue, err := metalInt32Scalar(pageSize, storageTensor.bridge)
	if err != nil {
		return metalPageConfig{}, err
	}

	config, err := pageConfigFromShapes(storageTensor, valueTensor, pageIDTensor, outTensor, uint32(pageSizeValue))
	if err != nil {
		return metalPageConfig{}, err
	}

	if !storageTensor.shape.Equal(outTensor.shape) || offsetTensor.dtype != dtype.Int32 ||
		offsetTensor.shape.Len() != pageIDTensor.shape.Len() {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	config.third = pageIDTensor
	config.offsets = offsetTensor

	return config, nil
}

func requireMetalPageGather(
	storage tensor.Tensor,
	pageTable tensor.Tensor,
	pageSize tensor.Tensor,
	out tensor.Tensor,
) (metalPageConfig, error) {
	storageTensor, pageTableTensor, outTensor, err := requireMetalShapeIndexBase(storage, pageTable, out)
	if err != nil {
		return metalPageConfig{}, err
	}

	pageSizeValue, err := metalInt32Scalar(pageSize, storageTensor.bridge)
	if err != nil {
		return metalPageConfig{}, err
	}

	storageDims := storageTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(storageDims) < 2 || len(outDims) != len(storageDims)-1 ||
		storageDims[1] != int(pageSizeValue) {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	inner, err := matchingTrailingProduct(outDims[1:], storageDims[2:])
	if err != nil {
		return metalPageConfig{}, err
	}

	if (outDims[0]+int(pageSizeValue)-1)/int(pageSizeValue) > pageTableTensor.shape.Len() {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(storageTensor.dtype)
	if err != nil {
		return metalPageConfig{}, err
	}

	return metalPageConfig{
		first:        storageTensor,
		second:       pageTableTensor,
		out:          outTensor,
		elementDType: elementDType,
		pageCount:    uint32(storageDims[0]),
		pageSize:     uint32(pageSizeValue),
		inner:        uint32(inner),
		rows:         uint32(outDims[0]),
	}, nil
}

func pageConfigFromShapes(
	storageTensor *metalTensor,
	valueTensor *metalTensor,
	pageIDTensor *metalTensor,
	outTensor *metalTensor,
	pageSize uint32,
) (metalPageConfig, error) {
	storageDims := storageTensor.shape.Dims()
	valueDims := valueTensor.shape.Dims()

	if len(storageDims) < 2 || len(valueDims) != len(storageDims)-1 ||
		storageDims[1] != int(pageSize) || valueDims[0] != pageIDTensor.shape.Len() {
		return metalPageConfig{}, tensor.ErrShapeMismatch
	}

	inner, err := matchingTrailingProduct(valueDims[1:], storageDims[2:])
	if err != nil {
		return metalPageConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(storageTensor.dtype)
	if err != nil {
		return metalPageConfig{}, err
	}

	if valueTensor.dtype != storageTensor.dtype || !storageTensor.shape.Equal(outTensor.shape) {
		return metalPageConfig{}, tensor.ErrDTypeMismatch
	}

	return metalPageConfig{
		first:        storageTensor,
		second:       valueTensor,
		out:          outTensor,
		elementDType: elementDType,
		pageCount:    uint32(storageDims[0]),
		pageSize:     pageSize,
		inner:        uint32(inner),
		rows:         uint32(valueDims[0]),
	}, nil
}

func matchingTrailingProduct(left []int, right []int) (int, error) {
	if len(left) != len(right) {
		return 0, tensor.ErrShapeMismatch
	}

	product := 1

	for index, dimension := range left {
		if dimension != right[index] {
			return 0, tensor.ErrShapeMismatch
		}

		product *= dimension
	}

	return product, nil
}
