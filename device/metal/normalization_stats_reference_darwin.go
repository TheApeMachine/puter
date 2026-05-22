//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

type normRowStats struct {
	mean      float32
	invStdDev float32
}

func metalGroupNormRowStats(
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

func metalInstanceNormRowStats(
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
