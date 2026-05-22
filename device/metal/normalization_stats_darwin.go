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
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

type normRowStats struct {
	mean      float32
	invStdDev float32
}

func metalGroupNormRowStatsForTest(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	batch int,
	channels int,
	spatial int,
	groups int,
) []normRowStats {
	testingObject.Helper()

	rowCount := batch * groups
	inputShape := mustShapeForTest(testingObject, []int{len(input)})
	statsShape := mustShapeForTest(testingObject, []int{rowCount})
	inputTensor, err := backend.Upload(inputShape, dtype.Float32, dtypeconvert.Float32ToBytes(input))
	if err != nil {
		testingObject.Fatalf("Upload groupnorm stats input failed: %v", err)
	}

	defer func() {
		if closeErr := inputTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close groupnorm stats input failed: %v", closeErr)
		}
	}()

	meanTensor := emptyTensorForTest(testingObject, backend, statsShape, dtype.Float32)
	invStdDevTensor := emptyTensorForTest(testingObject, backend, statsShape, dtype.Float32)

	defer func() {
		if closeErr := meanTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close groupnorm mean failed: %v", closeErr)
		}
	}()

	defer func() {
		if closeErr := invStdDevTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close groupnorm invStdDev failed: %v", closeErr)
		}
	}()

	if err := runMetalGroupNormStatsFloat32(
		context.Background(),
		inputTensor,
		meanTensor,
		invStdDevTensor,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		testingObject.Fatalf("runMetalGroupNormStatsFloat32 failed: %v", err)
	}

	means := downloadFloat32ForTest(testingObject, backend, meanTensor)
	invStdDevs := downloadFloat32ForTest(testingObject, backend, invStdDevTensor)
	stats := make([]normRowStats, rowCount)

	for rowIndex := range rowCount {
		stats[rowIndex] = normRowStats{
			mean:      means[rowIndex],
			invStdDev: invStdDevs[rowIndex],
		}
	}

	return stats
}

func metalInstanceNormRowStatsForTest(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	batch int,
	channels int,
	spatial int,
) []normRowStats {
	testingObject.Helper()

	rowCount := batch * channels
	inputShape := mustShapeForTest(testingObject, []int{len(input)})
	statsShape := mustShapeForTest(testingObject, []int{rowCount})
	inputTensor, err := backend.Upload(inputShape, dtype.Float32, dtypeconvert.Float32ToBytes(input))
	if err != nil {
		testingObject.Fatalf("Upload instancenorm stats input failed: %v", err)
	}

	defer func() {
		if closeErr := inputTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close instancenorm stats input failed: %v", closeErr)
		}
	}()

	meanTensor := emptyTensorForTest(testingObject, backend, statsShape, dtype.Float32)
	invStdDevTensor := emptyTensorForTest(testingObject, backend, statsShape, dtype.Float32)

	defer func() {
		if closeErr := meanTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close instancenorm mean failed: %v", closeErr)
		}
	}()

	defer func() {
		if closeErr := invStdDevTensor.Close(); closeErr != nil {
			testingObject.Fatalf("Close instancenorm invStdDev failed: %v", closeErr)
		}
	}()

	if err := runMetalInstanceNormStatsFloat32(
		context.Background(),
		inputTensor,
		meanTensor,
		invStdDevTensor,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
	); err != nil {
		testingObject.Fatalf("runMetalInstanceNormStatsFloat32 failed: %v", err)
	}

	means := downloadFloat32ForTest(testingObject, backend, meanTensor)
	invStdDevs := downloadFloat32ForTest(testingObject, backend, invStdDevTensor)
	stats := make([]normRowStats, rowCount)

	for rowIndex := range rowCount {
		stats[rowIndex] = normRowStats{
			mean:      means[rowIndex],
			invStdDev: invStdDevs[rowIndex],
		}
	}

	return stats
}

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
