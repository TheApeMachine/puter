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
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalActivationSteerConfig struct {
	destination  *metalTensor
	base         *metalTensor
	direction    *metalTensor
	coefficient  *metalTensor
	elementDType metalElementDType
	count        uint32
}

func runMetalActivationSteer(
	base tensor.Tensor,
	direction tensor.Tensor,
	coefficient tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalActivationSteer(base, direction, coefficient, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(
		config.destination,
		config.base,
		config.direction,
		config.coefficient,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_activation_steer(
		config.base.bridge.device,
		C.int(config.elementDType),
		config.destination.buffer,
		config.base.buffer,
		config.direction.buffer,
		config.coefficient.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("activation_steer", token, rc, status)
}

func requireMetalActivationSteer(
	base tensor.Tensor,
	direction tensor.Tensor,
	coefficient tensor.Tensor,
	out tensor.Tensor,
) (metalActivationSteerConfig, error) {
	tensors, err := requireMetalTensors(base, direction, coefficient, out)
	if err != nil {
		return metalActivationSteerConfig{}, err
	}

	baseTensor := tensors[0]
	directionTensor := tensors[1]
	coefficientTensor := tensors[2]
	destinationTensor := tensors[3]

	if baseTensor.bridge != directionTensor.bridge ||
		baseTensor.bridge != coefficientTensor.bridge ||
		baseTensor.bridge != destinationTensor.bridge {
		return metalActivationSteerConfig{}, errors.New(
			"metal activation steer: tensors belong to different Metal backends",
		)
	}

	elementDType, err := metalElementDTypeFor(baseTensor.dtype)
	if err != nil {
		return metalActivationSteerConfig{}, err
	}

	if baseTensor.dtype != directionTensor.dtype ||
		baseTensor.dtype != destinationTensor.dtype {
		return metalActivationSteerConfig{}, tensor.ErrDTypeMismatch
	}

	if coefficientTensor.dtype != dtype.Float32 {
		return metalActivationSteerConfig{}, tensor.ErrDTypeMismatch
	}

	if !baseTensor.shape.Equal(directionTensor.shape) ||
		!baseTensor.shape.Equal(destinationTensor.shape) {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	if coefficientTensor.shape.Len() != 1 {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	if baseTensor.shape.Len() > math.MaxUint32 {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	return metalActivationSteerConfig{
		destination:  destinationTensor,
		base:         baseTensor,
		direction:    directionTensor,
		coefficient:  coefficientTensor,
		elementDType: elementDType,
		count:        uint32(baseTensor.shape.Len()),
	}, nil
}
