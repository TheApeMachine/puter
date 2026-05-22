//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalSwiGLUConfig struct {
	destination *metalTensor
	gate        *metalTensor
	up          *metalTensor
	count       uint32
}

type metalPackedSwiGLUConfig struct {
	destination *metalTensor
	packed      *metalTensor
	inner       uint32
	count       uint32
}

func runMetalSwiGLU(gate tensor.Tensor, up tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalSwiGLU(gate, up, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	elementDType, err := metalElementDTypeFor(config.gate.dtype)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.destination, config.gate, config.up)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_swiglu(
		config.gate.bridge.device,
		C.int(elementDType),
		config.destination.buffer,
		config.gate.buffer,
		config.up.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal swiglu: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalPackedSwiGLU(packed tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalPackedSwiGLU(packed, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	elementDType, err := metalElementDTypeFor(config.packed.dtype)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.destination, config.packed)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_swiglu_packed(
		config.packed.bridge.device,
		C.int(elementDType),
		config.destination.buffer,
		config.packed.buffer,
		C.uint32_t(config.inner),
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal packed swiglu: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalSwiGLU(
	gate tensor.Tensor,
	up tensor.Tensor,
	out tensor.Tensor,
) (metalSwiGLUConfig, error) {
	tensors, err := requireMetalTensors(gate, up, out)
	if err != nil {
		return metalSwiGLUConfig{}, err
	}

	gateTensor := tensors[0]
	upTensor := tensors[1]
	destinationTensor := tensors[2]

	if gateTensor.bridge != upTensor.bridge ||
		gateTensor.bridge != destinationTensor.bridge {
		return metalSwiGLUConfig{}, errors.New(
			"metal swiglu: tensors belong to different Metal backends",
		)
	}

	if gateTensor.dtype != upTensor.dtype ||
		gateTensor.dtype != destinationTensor.dtype {
		return metalSwiGLUConfig{}, tensor.ErrDTypeMismatch
	}

	if !gateTensor.shape.Equal(upTensor.shape) ||
		!gateTensor.shape.Equal(destinationTensor.shape) {
		return metalSwiGLUConfig{}, tensor.ErrShapeMismatch
	}

	if gateTensor.shape.Len() > math.MaxUint32 {
		return metalSwiGLUConfig{}, tensor.ErrShapeMismatch
	}

	return metalSwiGLUConfig{
		destination: destinationTensor,
		gate:        gateTensor,
		up:          upTensor,
		count:       uint32(gateTensor.shape.Len()),
	}, nil
}

func requireMetalPackedSwiGLU(
	packed tensor.Tensor,
	out tensor.Tensor,
) (metalPackedSwiGLUConfig, error) {
	tensors, err := requireMetalTensors(packed, out)
	if err != nil {
		return metalPackedSwiGLUConfig{}, err
	}

	packedTensor := tensors[0]
	destinationTensor := tensors[1]

	if packedTensor.bridge != destinationTensor.bridge {
		return metalPackedSwiGLUConfig{}, errors.New(
			"metal packed swiglu: tensors belong to different Metal backends",
		)
	}

	if packedTensor.dtype != destinationTensor.dtype {
		return metalPackedSwiGLUConfig{}, tensor.ErrDTypeMismatch
	}

	packedDims := packedTensor.shape.Dims()
	outDims := destinationTensor.shape.Dims()

	if len(packedDims) == 0 || len(packedDims) != len(outDims) {
		return metalPackedSwiGLUConfig{}, tensor.ErrShapeMismatch
	}

	lastIndex := len(packedDims) - 1

	if packedDims[lastIndex] != outDims[lastIndex]*2 {
		return metalPackedSwiGLUConfig{}, tensor.ErrShapeMismatch
	}

	for index := 0; index < lastIndex; index++ {
		if packedDims[index] != outDims[index] {
			return metalPackedSwiGLUConfig{}, tensor.ErrShapeMismatch
		}
	}

	if destinationTensor.shape.Len() > math.MaxUint32 {
		return metalPackedSwiGLUConfig{}, tensor.ErrShapeMismatch
	}

	return metalPackedSwiGLUConfig{
		destination: destinationTensor,
		packed:      packedTensor,
		inner:       uint32(outDims[lastIndex]),
		count:       uint32(destinationTensor.shape.Len()),
	}, nil
}
