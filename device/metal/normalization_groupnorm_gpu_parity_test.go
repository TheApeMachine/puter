package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestGroupNormGPUVersusSerialReference(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	spatial := 7
	batch, channels := norm3DShape()
	groups := metalDefaultGroupNormGroups
	fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
	input, scale, bias, out := norm3DAffineTensorsForTest(
		t, backend, dtype.Float32, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, out)

	err := lookupNorm3DKernel(t, "groupnorm", dtype.Float32).Run(input, scale, bias, out)
	if err != nil {
		t.Fatalf("groupnorm Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	serial := groupNormMetalSerialReference(
		inputStored, scaleStored, biasStored, batch, channels, spatial, groups,
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
