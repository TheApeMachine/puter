//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalTimestepConfig struct {
	timesteps *metalTensor
	maxPeriod *metalTensor
	downscale *metalTensor
	flip      *metalTensor
	out       *metalTensor
	element   metalElementDType
	count     uint32
	dim       uint32
}

func runMetalTimestep(
	timesteps tensor.Tensor,
	maxPeriod tensor.Tensor,
	downscale tensor.Tensor,
	flip tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalTimestep(timesteps, maxPeriod, downscale, flip, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(
		config.out,
		config.timesteps,
		config.maxPeriod,
		config.downscale,
		config.flip,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_timestep_embedding(
		config.timesteps.bridge.device,
		config.timesteps.buffer,
		config.maxPeriod.buffer,
		config.downscale.buffer,
		config.flip.buffer,
		config.out.buffer,
		C.int(config.element),
		C.uint32_t(config.count),
		C.uint32_t(config.dim),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal timestep: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalTimestep(
	timesteps tensor.Tensor,
	maxPeriod tensor.Tensor,
	downscale tensor.Tensor,
	flip tensor.Tensor,
	out tensor.Tensor,
) (metalTimestepConfig, error) {
	tensors, err := requireMetalTensors(timesteps, maxPeriod, downscale, flip, out)
	if err != nil {
		return metalTimestepConfig{}, err
	}

	config := metalTimestepConfig{
		timesteps: tensors[0],
		maxPeriod: tensors[1],
		downscale: tensors[2],
		flip:      tensors[3],
		out:       tensors[4],
	}

	if err := requireMetalTimestepTypes(config); err != nil {
		return metalTimestepConfig{}, err
	}

	element, err := metalElementDTypeFor(config.out.dtype)
	if err != nil {
		return metalTimestepConfig{}, err
	}

	count := config.timesteps.shape.Len()

	if count <= 0 || config.out.shape.Len()%count != 0 {
		return metalTimestepConfig{}, tensor.ErrShapeMismatch
	}

	dim := config.out.shape.Len() / count

	if dim <= 0 {
		return metalTimestepConfig{}, tensor.ErrShapeMismatch
	}

	config.count = uint32(count)
	config.dim = uint32(dim)
	config.element = element

	return config, requireUint32(config.out.shape.Len())
}

func requireMetalTimestepTypes(config metalTimestepConfig) error {
	if config.timesteps.dtype != dtype.Float32 ||
		config.maxPeriod.dtype != dtype.Float32 ||
		config.downscale.dtype != dtype.Float32 {
		return tensor.ErrDTypeMismatch
	}

	if config.flip.dtype != dtype.Int32 {
		return tensor.ErrDTypeMismatch
	}

	if config.maxPeriod.shape.Len() != 1 ||
		config.downscale.shape.Len() != 1 ||
		config.flip.shape.Len() != 1 {
		return tensor.ErrShapeMismatch
	}

	bridge := config.timesteps.bridge

	if config.maxPeriod.bridge != bridge ||
		config.downscale.bridge != bridge ||
		config.flip.bridge != bridge ||
		config.out.bridge != bridge {
		return tensor.ErrShapeMismatch
	}

	return nil
}
