//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <stdlib.h>
#include "bridge_darwin.h"
*/
import "C"

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalUnaryNamedFloat32(
	ctx context.Context,
	kernelName string,
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) error {
	inputMetal, outMetal, err := requireMetalUnaryNamedFloat32Tensors(inputTensor, outTensor)
	if err != nil {
		return err
	}

	if inputMetal.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outMetal, inputMetal)
	if err != nil {
		return err
	}

	kernelCString := C.CString(kernelName)
	defer C.free(unsafe.Pointer(kernelCString))

	status := C.MetalStatus{}
	rc := C.metal_dispatch_unary_named_float32(
		inputMetal.bridge.device,
		kernelCString,
		inputMetal.buffer,
		outMetal.buffer,
		C.uint32_t(inputMetal.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal %s: %s", kernelName, metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalUnaryNamedFloat32Tensors(
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputMetal, err := requireMetalTensor(inputTensor)
	if err != nil {
		return nil, nil, err
	}

	outMetal, err := requireMetalTensor(outTensor)
	if err != nil {
		return nil, nil, err
	}

	if inputMetal.dtype != dtype.Float32 || outMetal.dtype != dtype.Float32 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if !inputMetal.shape.Equal(outMetal.shape) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if inputMetal.bridge != outMetal.bridge {
		return nil, nil, fmt.Errorf("metal unary_named_float32: tensors belong to different Metal backends")
	}

	return inputMetal, outMetal, nil
}

func runMetalFMAFloat32(
	ctx context.Context,
	aTensor tensor.Tensor,
	bTensor tensor.Tensor,
	cTensor tensor.Tensor,
	outTensor tensor.Tensor,
) error {
	aMetal, bMetal, cMetal, outMetal, err := requireMetalFMAFloat32Tensors(aTensor, bTensor, cTensor, outTensor)
	if err != nil {
		return err
	}

	if aMetal.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outMetal, aMetal, bMetal, cMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_fma_float32(
		aMetal.bridge.device,
		aMetal.buffer,
		bMetal.buffer,
		cMetal.buffer,
		outMetal.buffer,
		C.uint32_t(aMetal.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal fma_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalFMAFloat32Tensors(
	aTensor tensor.Tensor,
	bTensor tensor.Tensor,
	cTensor tensor.Tensor,
	outTensor tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, *metalTensor, error) {
	aMetal, err := requireMetalTensor(aTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	bMetal, err := requireMetalTensor(bTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	cMetal, err := requireMetalTensor(cTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	outMetal, err := requireMetalTensor(outTensor)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	if aMetal.dtype != dtype.Float32 ||
		bMetal.dtype != dtype.Float32 ||
		cMetal.dtype != dtype.Float32 ||
		outMetal.dtype != dtype.Float32 {
		return nil, nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if !aMetal.shape.Equal(bMetal.shape) ||
		!aMetal.shape.Equal(cMetal.shape) ||
		!aMetal.shape.Equal(outMetal.shape) {
		return nil, nil, nil, nil, tensor.ErrShapeMismatch
	}

	if aMetal.bridge != bMetal.bridge ||
		aMetal.bridge != cMetal.bridge ||
		aMetal.bridge != outMetal.bridge {
		return nil, nil, nil, nil, fmt.Errorf("metal fma_float32: tensors belong to different Metal backends")
	}

	return aMetal, bMetal, cMetal, outMetal, nil
}

func runMetalInvStdDevFloat32(
	ctx context.Context,
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) error {
	inputMetal, outMetal, err := requireMetalInvStdDevFloat32Tensors(inputTensor, outTensor)
	if err != nil {
		return err
	}

	if inputMetal.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outMetal, inputMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_inv_std_dev_float32(
		inputMetal.bridge.device,
		inputMetal.buffer,
		outMetal.buffer,
		C.uint32_t(inputMetal.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal inv_std_dev_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalInvStdDevFloat32Tensors(
	inputTensor tensor.Tensor,
	outTensor tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputMetal, err := requireMetalTensor(inputTensor)
	if err != nil {
		return nil, nil, err
	}

	outMetal, err := requireMetalTensor(outTensor)
	if err != nil {
		return nil, nil, err
	}

	if inputMetal.dtype != dtype.Float32 || outMetal.dtype != dtype.Float32 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if !inputMetal.shape.Equal(outMetal.shape) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if inputMetal.bridge != outMetal.bridge {
		return nil, nil, fmt.Errorf("metal inv_std_dev_float32: tensors belong to different Metal backends")
	}

	return inputMetal, outMetal, nil
}
