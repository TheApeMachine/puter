package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestInstanceNormGPUVersusSerialReference(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	spatial := 7
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
	input, scale, bias, out := norm3DAffineTensorsForTest(
		t, backend, dtype.Float32, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupNorm3DKernel(t, "instancenorm", dtype.Float32).Run(input, scale, bias, out)
	if err != nil {
		t.Fatalf("instancenorm Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	serial := instanceNormMetalSerialReference(
		inputStored, scaleStored, biasStored, batch, channels, spatial,
	)
	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, serial)
	if maxDistance > normalizationNorm3DFloat32MaxULP {
		t.Fatalf(
			"GPU vs serial at %d: got %08x (%g), want %08x (%g), distance %d > %d",
			maxIndex,
			math.Float32bits(actual[maxIndex]),
			actual[maxIndex],
			math.Float32bits(serial[maxIndex]),
			serial[maxIndex],
			maxDistance,
			normalizationNorm3DFloat32MaxULP,
		)
	}
}

func instanceNormMetalSerialReference(
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			mean := normalizationMeanForTest(row)
			variance := normalizationVarianceForTest(row, mean)
			invStdDev := normInvStdDev(variance)
			applyNorm3DExpectedRow(
				row, out[start:start+spatial],
				scale[channelIndex], bias[channelIndex], mean, invStdDev,
			)
		}
	}

	return out
}
