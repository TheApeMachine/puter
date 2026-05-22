package metal

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestNorm3DULPProbeFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	batch, channels := norm3DShape()

	for _, spatial := range parityElementCounts {
		spatial := spatial

		t.Run(fmt.Sprintf("spatial=%d", spatial), func(t *testing.T) {
			probeNorm3DOpULP(t, backend, "groupnorm", batch, channels, spatial)
			probeNorm3DOpULP(t, backend, "instancenorm", batch, channels, spatial)
			probeNorm3DOpULP(t, backend, "batchnorm_eval", batch, channels, spatial)
		})
	}
}

func probeNorm3DOpULP(
	testingObject *testing.T,
	backend *Backend,
	opName string,
	batch int,
	channels int,
	spatial int,
) {
	testingObject.Helper()

	fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
	expectedBytes := norm3DExpectedBytesForTest(
		testingObject, backend, dtype.Float32, fixture, batch, channels, spatial, opName,
	)
	expected := decodeDTypeBytesToFloat32(expectedBytes, dtype.Float32)

	var actual []float32

	switch opName {
	case "groupnorm":
		input, scale, bias, out := norm3DAffineTensorsForTest(
			testingObject, backend, dtype.Float32, batch, channels, spatial, fixture,
		)
		defer closeBenchmarkTensors(input, scale, bias, out)

		err := lookupNorm3DKernel(testingObject, "groupnorm", dtype.Float32).Run(input, scale, bias, out)
		if err != nil {
			testingObject.Fatalf("groupnorm Run failed: %v", err)
		}

		_, actualBytes, err := backend.Download(out)
		if err != nil {
			testingObject.Fatalf("Download failed: %v", err)
		}

		actual = decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	case "instancenorm":
		input, scale, bias, out := norm3DAffineTensorsForTest(
			testingObject, backend, dtype.Float32, batch, channels, spatial, fixture,
		)
		defer closeBenchmarkTensors(input, scale, bias, out)

		err := lookupNorm3DKernel(testingObject, "instancenorm", dtype.Float32).Run(input, scale, bias, out)
		if err != nil {
			testingObject.Fatalf("instancenorm Run failed: %v", err)
		}

		_, actualBytes, err := backend.Download(out)
		if err != nil {
			testingObject.Fatalf("Download failed: %v", err)
		}

		actual = decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	case "batchnorm_eval":
		input, scale, bias, mean, variance, out := batchNormEvalTensorsForTest(
			testingObject, backend, dtype.Float32, batch, channels, spatial, fixture,
		)
		defer closeBenchmarkTensors(input, scale, bias, mean, variance, out)

		err := lookupBatchNormEvalKernel(testingObject, dtype.Float32).Run(input, scale, bias, mean, variance, out)
		if err != nil {
			testingObject.Fatalf("batchnorm_eval Run failed: %v", err)
		}

		_, actualBytes, err := backend.Download(out)
		if err != nil {
			testingObject.Fatalf("Download failed: %v", err)
		}

		actual = decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	default:
		testingObject.Fatalf("unknown op: %s", opName)
	}

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actual, expected)
	testingObject.Logf(
		"%s spatial=%d maxULP=%d at %d got=%g want=%g",
		opName,
		spatial,
		maxDistance,
		maxIndex,
		actual[maxIndex],
		expected[maxIndex],
	)
}
