//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestBatchNormEvalGPUVersusSerialReference(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	spatial := 1024
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
	input, scale, bias, mean, variance, out := batchNormEvalTensorsForTest(
		t, backend, dtype.Float32, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, mean, variance, out)

	err := lookupBatchNormEvalKernel(t, dtype.Float32).Run(input, scale, bias, mean, variance, out)
	if err != nil {
		t.Fatalf("batchnorm_eval Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	meanStored := decodeDTypeBytesToFloat32(fixture.meanBytes, dtype.Float32)
	varianceStored := decodeDTypeBytesToFloat32(fixture.varianceBytes, dtype.Float32)
	serial := expectedBatchNormEvalValuesMetalSqrt(
		t, backend,
		inputStored, scaleStored, biasStored, meanStored, varianceStored,
		batch, channels, spatial,
	)
	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, serial)
	if maxDistance > normalizationNorm3DMaxULP("batchnorm_eval") {
		t.Fatalf(
			"GPU vs serial at %d: got %08x (%g), want %08x (%g), distance %d > %d",
			maxIndex,
			math.Float32bits(actual[maxIndex]),
			actual[maxIndex],
			math.Float32bits(serial[maxIndex]),
			serial[maxIndex],
			maxDistance,
			normalizationNorm3DMaxULP("batchnorm_eval"),
		)
	}
}

func batchNormEvalMetalSerialReference(
	input []float32,
	scale []float32,
	bias []float32,
	mean []float32,
	variance []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			start := (batchIndex*channels + channelIndex) * spatial
			invStdDev := normInvStdDev(variance[channelIndex])
			applyNorm3DExpectedRow(
				input[start:start+spatial],
				out[start:start+spatial],
				scale[channelIndex],
				bias[channelIndex],
				mean[channelIndex],
				invStdDev,
			)
		}
	}

	return out
}
