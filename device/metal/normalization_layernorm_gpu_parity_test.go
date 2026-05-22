package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestLayerNormGPUVersusSerialReferenceN7(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	rows, cols := 2, 7
	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, dtype.Float32)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, dtype.Float32)
	serial := layerNormMetalSerialReference(inputStored, scaleStored, biasStored, rows, cols)

	input, scale, bias, out := layerNormTensorsForTest(
		t, backend, rows, cols, dtype.Float32, inputBytes, scaleBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupLayerNormKernel(t, dtype.Float32).Run(input, scale, bias, out)
	if err != nil {
		t.Fatalf("layernorm Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, serial)
	if maxDistance > 32 {
		t.Fatalf(
			"GPU vs serial at %d: got %08x (%g), want %08x (%g), distance %d > 32",
			maxIndex,
			math.Float32bits(actual[maxIndex]),
			actual[maxIndex],
			math.Float32bits(serial[maxIndex]),
			serial[maxIndex],
			maxDistance,
		)
	}
}

func TestLayerNormGPUVersusSerialReferenceCols14NormInput(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	rows, cols := 1, 14
	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, dtype.Float32)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, dtype.Float32)
	serial := layerNormMetalSerialReference(inputStored, scaleStored, biasStored, rows, cols)

	input, scale, bias, out := layerNormTensorsForTest(
		t, backend, rows, cols, dtype.Float32, inputBytes, scaleBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupLayerNormKernel(t, dtype.Float32).Run(input, scale, bias, out)
	if err != nil {
		t.Fatalf("layernorm Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, serial)
	if maxDistance > 32 {
		t.Fatalf(
			"normInput cols=14 GPU vs serial at %d: distance %d > 32",
			maxIndex, maxDistance,
		)
	}
}

func TestLayerNormGPUVersusSerialReferenceGroupSliceN7(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	spatial := 7
	batch, channels := norm3DShape()
	groups := metalDefaultGroupNormGroups
	channelsPerGroup := channels / groups
	input, scale, bias, _, _ := norm3DValues(batch, channels, spatial)

	batchIndex := 1
	groupIndex := 3
	channelStart := groupIndex * channelsPerGroup
	groupStart := (batchIndex*channels + channelStart) * spatial
	groupSize := channelsPerGroup * spatial
	rowInput := append([]float32(nil), input[groupStart:groupStart+groupSize]...)
	rowScale := make([]float32, groupSize)
	rowBias := make([]float32, groupSize)

	for channelIndex := range channelsPerGroup {
		for spatialIndex := range spatial {
			index := channelIndex*spatial + spatialIndex
			channel := channelStart + channelIndex
			rowScale[index] = scale[channel]
			rowBias[index] = bias[channel]
		}
	}

	serial := layerNormMetalSerialReference(rowInput, rowScale, rowBias, 1, groupSize)
	inputBytes := encodeNormValuesAsDType(rowInput, dtype.Float32)
	scaleBytes := encodeNormValuesAsDType(rowScale, dtype.Float32)
	biasBytes := encodeNormValuesAsDType(rowBias, dtype.Float32)

	inputTensor, scaleTensor, biasTensor, out := layerNormTensorsForTest(
		t, backend, 1, groupSize, dtype.Float32, inputBytes, scaleBytes, biasBytes,
	)
	defer closeBenchmarkTensors(inputTensor, scaleTensor, biasTensor, out)

	err := lookupLayerNormKernel(t, dtype.Float32).Run(inputTensor, scaleTensor, biasTensor, out)
	if err != nil {
		t.Fatalf("layernorm Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, serial)
	if maxDistance > 32 {
		t.Fatalf(
			"group slice GPU vs serial at %d: got %08x (%g), want %08x (%g), distance %d > 32",
			maxIndex,
			math.Float32bits(actual[maxIndex]),
			actual[maxIndex],
			math.Float32bits(serial[maxIndex]),
			serial[maxIndex],
			maxDistance,
		)
	}
}

func layerNormMetalSerialReference(
	input []float32,
	scale []float32,
	bias []float32,
	rows int,
	cols int,
) []float32 {
	out := make([]float32, len(input))

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		row := input[rowOffset : rowOffset+cols]
		mean := normalizationMeanForTest(row)
		variance := normalizationVarianceForTest(row, mean)
		invStdDev := normInvStdDev(variance)
		applyLayerNormExpectedRowForTest(
			row, out[rowOffset:rowOffset+cols], scale, bias, mean, invStdDev,
		)
	}

	return out
}
