//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

func metalSqrtFloat32ForTest(
	testingObject testing.TB,
	backend *Backend,
	values []float32,
) []float32 {
	testingObject.Helper()

	if len(values) == 0 {
		return nil
	}

	shape := mustShapeForTest(testingObject, []int{len(values)})
	input, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(values))
	if err != nil {
		testingObject.Fatalf("Upload sqrt input failed: %v", err)
	}

	defer func() {
		if closeErr := input.Close(); closeErr != nil {
			testingObject.Fatalf("Close sqrt input failed: %v", closeErr)
		}
	}()

	output, err := backend.SqrtFloat32(context.Background(), input)
	if err != nil {
		testingObject.Fatalf("SqrtFloat32 failed: %v", err)
	}

	defer func() {
		if closeErr := output.Close(); closeErr != nil {
			testingObject.Fatalf("Close sqrt output failed: %v", closeErr)
		}
	}()

	return downloadFloat32ForTest(testingObject, backend, output)
}

func metalRecipFloat32ForTest(
	testingObject testing.TB,
	backend *Backend,
	values []float32,
) []float32 {
	testingObject.Helper()

	if len(values) == 0 {
		return nil
	}

	shape := mustShapeForTest(testingObject, []int{len(values)})
	input, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(values))
	if err != nil {
		testingObject.Fatalf("Upload recip input failed: %v", err)
	}

	defer func() {
		if closeErr := input.Close(); closeErr != nil {
			testingObject.Fatalf("Close recip input failed: %v", closeErr)
		}
	}()

	output, err := backend.RecipFloat32(context.Background(), input)
	if err != nil {
		testingObject.Fatalf("RecipFloat32 failed: %v", err)
	}

	defer func() {
		if closeErr := output.Close(); closeErr != nil {
			testingObject.Fatalf("Close recip output failed: %v", closeErr)
		}
	}()

	return downloadFloat32ForTest(testingObject, backend, output)
}

func metalInvStdDevsForTest(
	testingObject testing.TB,
	backend *Backend,
	variancePlusEpsilon []float32,
) []float32 {
	testingObject.Helper()

	return metalInvStdDevPreciseFloat32(testingObject, backend, variancePlusEpsilon)
}

func expectedBatchNormEvalValuesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	scale []float32,
	bias []float32,
	mean []float32,
	variance []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	testingObject.Helper()

	variancePlusEpsilon := make([]float32, channels)

	for channelIndex := range channels {
		variancePlusEpsilon[channelIndex] = variance[channelIndex] + layerNormEpsilonMetalForTest
	}

	invStdDevs := metalInvStdDevsForTest(testingObject, backend, variancePlusEpsilon)
	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			start := (batchIndex*channels + channelIndex) * spatial
			applyNorm3DExpectedRowGPU(
				testingObject,
				backend,
				input[start:start+spatial],
				out[start:start+spatial],
				scale[channelIndex],
				bias[channelIndex],
				mean[channelIndex],
				invStdDevs[channelIndex],
			)
		}
	}

	return out
}

func expectedGroupNormValuesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	testingObject.Helper()

	groups := metalDefaultGroupNormGroups
	channelsPerGroup := channels / groups
	groupStats := metalGroupNormRowStats(
		testingObject, backend, input, batch, channels, spatial, groups,
	)

	out := make([]float32, len(input))
	statsIndex := 0

	for batchIndex := range batch {
		for groupIndex := range groups {
			channelStart := groupIndex * channelsPerGroup
			groupStart := (batchIndex*channels + channelStart) * spatial
			mean := groupStats[statsIndex].mean
			invStdDev := groupStats[statsIndex].invStdDev
			statsIndex++

			groupSize := channelsPerGroup * spatial
			scaleByElement := make([]float32, groupSize)
			biasByElement := make([]float32, groupSize)

			for channelIndex := range channelsPerGroup {
				channel := channelStart + channelIndex
				for spatialIndex := range spatial {
					index := channelIndex*spatial + spatialIndex
					scaleByElement[index] = scale[channel]
					biasByElement[index] = bias[channel]
				}
			}

			applyNorm3DAffineSliceGPU(
				testingObject,
				backend,
				input[groupStart:groupStart+groupSize],
				out[groupStart:groupStart+groupSize],
				scaleByElement,
				biasByElement,
				mean,
				invStdDev,
			)
		}
	}

	return out
}

func expectedInstanceNormValuesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	testingObject.Helper()

	rowStats := metalInstanceNormRowStats(
		testingObject, backend, input, batch, channels, spatial,
	)

	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			rowIndex := batchIndex*channels + channelIndex
			start := rowIndex * spatial
			scaleByElement := make([]float32, spatial)
			biasByElement := make([]float32, spatial)

			for spatialIndex := range spatial {
				scaleByElement[spatialIndex] = scale[channelIndex]
				biasByElement[spatialIndex] = bias[channelIndex]
			}

			applyNorm3DAffineSliceGPU(
				testingObject,
				backend,
				input[start:start+spatial],
				out[start:start+spatial],
				scaleByElement,
				biasByElement,
				rowStats[rowIndex].mean,
				rowStats[rowIndex].invStdDev,
			)
		}
	}

	return out
}

func norm3DFixtureBatchBytesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	fixture norm3DFixture,
	batch int,
	channels int,
	spatial int,
) []byte {
	testingObject.Helper()

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	meanStored := decodeDTypeBytesToFloat32(fixture.meanBytes, dtype.Float32)
	varianceStored := decodeDTypeBytesToFloat32(fixture.varianceBytes, dtype.Float32)
	expectedValues := expectedBatchNormEvalValuesMetalSqrt(
		testingObject, backend,
		inputStored, scaleStored, biasStored, meanStored, varianceStored,
		batch, channels, spatial,
	)

	return encodeNormValuesAsDType(expectedValues, dtype.Float32)
}

func norm3DFixtureGroupBytesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	fixture norm3DFixture,
	batch int,
	channels int,
	spatial int,
) []byte {
	testingObject.Helper()

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	expectedValues := expectedGroupNormValuesMetalSqrt(
		testingObject, backend,
		inputStored, scaleStored, biasStored,
		batch, channels, spatial,
	)

	return encodeNormValuesAsDType(expectedValues, dtype.Float32)
}

func norm3DExpectedBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	fixture norm3DFixture,
	batch int,
	channels int,
	spatial int,
	opName string,
) []byte {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		switch opName {
		case "groupnorm":
			return fixture.groupBytes
		case "instancenorm":
			return fixture.instanceBytes
		case "batchnorm_eval":
			return fixture.batchBytes
		default:
			testingObject.Fatalf("unknown norm3D op: %s", opName)
		}
	}

	switch opName {
	case "groupnorm":
		return norm3DFixtureGroupBytesMetalSqrt(testingObject, backend, fixture, batch, channels, spatial)
	case "instancenorm":
		return norm3DFixtureInstanceBytesMetalSqrt(testingObject, backend, fixture, batch, channels, spatial)
	case "batchnorm_eval":
		return norm3DFixtureBatchBytesMetalSqrt(testingObject, backend, fixture, batch, channels, spatial)
	default:
		testingObject.Fatalf("unknown norm3D op: %s", opName)
	}

	return nil
}

func norm3DFixtureInstanceBytesMetalSqrt(
	testingObject testing.TB,
	backend *Backend,
	fixture norm3DFixture,
	batch int,
	channels int,
	spatial int,
) []byte {
	testingObject.Helper()

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	expectedValues := expectedInstanceNormValuesMetalSqrt(
		testingObject, backend,
		inputStored, scaleStored, biasStored,
		batch, channels, spatial,
	)

	return encodeNormValuesAsDType(expectedValues, dtype.Float32)
}
