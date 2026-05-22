//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalGroupNormStatsFloat32(
	ctx context.Context,
	inputTensor tensor.Tensor,
	meanTensor tensor.Tensor,
	invStdDevTensor tensor.Tensor,
	batch uint32,
	channels uint32,
	spatial uint32,
	groups uint32,
) error {
	inputMetal, meanMetal, invStdDevMetal, err := requireMetalNormStatsTensors(inputTensor, meanTensor, invStdDevTensor)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(meanMetal, invStdDevMetal, inputMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_groupnorm_stats_float32(
		inputMetal.bridge.device,
		inputMetal.buffer,
		meanMetal.buffer,
		invStdDevMetal.buffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		C.uint32_t(groups),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal groupnorm_stats_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func runMetalInstanceNormStatsFloat32(
	ctx context.Context,
	inputTensor tensor.Tensor,
	meanTensor tensor.Tensor,
	invStdDevTensor tensor.Tensor,
	batch uint32,
	channels uint32,
	spatial uint32,
) error {
	inputMetal, meanMetal, invStdDevMetal, err := requireMetalNormStatsTensors(inputTensor, meanTensor, invStdDevTensor)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(meanMetal, invStdDevMetal, inputMetal)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_instancenorm_stats_float32(
		inputMetal.bridge.device,
		inputMetal.buffer,
		meanMetal.buffer,
		invStdDevMetal.buffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		dispatchErr := fmt.Errorf("metal instancenorm_stats_float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, dispatchErr)

		return dispatchErr
	}

	_ = ctx

	return nil
}

func requireMetalNormStatsTensors(
	inputTensor tensor.Tensor,
	meanTensor tensor.Tensor,
	invStdDevTensor tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	inputMetal, err := requireMetalTensor(inputTensor)
	if err != nil {
		return nil, nil, nil, err
	}

	meanMetal, err := requireMetalTensor(meanTensor)
	if err != nil {
		return nil, nil, nil, err
	}

	invStdDevMetal, err := requireMetalTensor(invStdDevTensor)
	if err != nil {
		return nil, nil, nil, err
	}

	if inputMetal.dtype != dtype.Float32 ||
		meanMetal.dtype != dtype.Float32 ||
		invStdDevMetal.dtype != dtype.Float32 {
		return nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if !meanMetal.shape.Equal(invStdDevMetal.shape) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if inputMetal.bridge != meanMetal.bridge || inputMetal.bridge != invStdDevMetal.bridge {
		return nil, nil, nil, fmt.Errorf("metal norm stats: tensors belong to different Metal backends")
	}

	return inputMetal, meanMetal, invStdDevMetal, nil
}
