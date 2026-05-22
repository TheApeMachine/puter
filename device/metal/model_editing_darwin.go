//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalWeightGraftConfig struct {
	weights      *metalTensor
	injection    *metalTensor
	elementDType metalElementDType
	count        uint32
}

func runMetalWeightGraftAdd(weights tensor.Tensor, injection tensor.Tensor) error {
	config, err := requireMetalWeightGraft(weights, injection)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.weights, config.injection)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_weight_graft_add(
		config.weights.bridge.device,
		C.int(config.elementDType),
		config.weights.buffer,
		config.injection.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("weight_graft_add", token, rc, status)
}

func requireMetalWeightGraft(
	weights tensor.Tensor,
	injection tensor.Tensor,
) (metalWeightGraftConfig, error) {
	weightTensor, injectionTensor, err := requireMetalUtilityTensors(weights, injection)
	if err != nil {
		return metalWeightGraftConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(weightTensor.dtype)
	if err != nil {
		return metalWeightGraftConfig{}, err
	}

	if weightTensor.dtype != injectionTensor.dtype {
		return metalWeightGraftConfig{}, tensor.ErrDTypeMismatch
	}

	if !weightTensor.shape.Equal(injectionTensor.shape) {
		return metalWeightGraftConfig{}, tensor.ErrShapeMismatch
	}

	if weightTensor.shape.Len() > math.MaxUint32 {
		return metalWeightGraftConfig{}, tensor.ErrShapeMismatch
	}

	return metalWeightGraftConfig{
		weights:      weightTensor,
		injection:    injectionTensor,
		elementDType: elementDType,
		count:        uint32(weightTensor.shape.Len()),
	}, nil
}
