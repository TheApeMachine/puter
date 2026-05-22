//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

func TestMetalAdaptiveRMSNorm(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given input rows and scale/shift modulation", t, func() {
		inputShape := mustShapeForTest(t, []int{1, 2, 3})
		modulationShape := mustShapeForTest(t, []int{1, 6})
		outputShape := mustShapeForTest(t, []int{1, 2, 3})
		input := uploadDTypeTensorForTest(
			t,
			backend,
			inputShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{1, 2, 3, 2, 4, 6}),
		)
		modulation := uploadDTypeTensorForTest(
			t,
			backend,
			modulationShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{0.5, 1, -0.5, 1, -1, 0.25}),
		)
		output := emptyTensorForTest(t, backend, outputShape, dtype.Float32)

		defer closeBenchmarkTensors(input, modulation, output)

		convey.Convey("It should normalize each row and broadcast modulation by batch", func() {
			err := runMetalAdaptiveRMSNorm(input, modulation, output)

			convey.So(err, convey.ShouldBeNil)
			assertFloat32WithinULP(
				t,
				mustDownloadFloat32ForTest(t, output),
				expectedAdaptiveRMSNormValues(),
				32,
			)
		})
	})
}

func expectedAdaptiveRMSNormValues() []float32 {
	input := []float32{1, 2, 3, 2, 4, 6}
	scale := []float32{0.5, 1, -0.5}
	shift := []float32{1, -1, 0.25}
	output := make([]float32, len(input))

	for row := range 2 {
		rowOffset := row * 3
		sum := float32(0)

		for col := range 3 {
			value := input[rowOffset+col]
			sum += value * value
		}

		invRMS := 1 / float32(math.Sqrt(float64(sum/3+rmsNormEpsilonMetalForTest)))

		for col := range 3 {
			normalized := input[rowOffset+col] * invRMS
			output[rowOffset+col] = normalized*(1+scale[col]) + shift[col]
		}
	}

	return output
}

func mustDownloadFloat32ForTest(t testing.TB, value any) []float32 {
	t.Helper()

	tensorValue, ok := value.(interface {
		RawBytes() (dtype.DType, []byte, error)
	})

	if !ok {
		t.Fatalf("value cannot expose raw bytes")
	}

	actualDType, rawBytes, err := tensorValue.RawBytes()

	if err != nil {
		t.Fatalf("RawBytes failed: %v", err)
	}

	if actualDType != dtype.Float32 {
		t.Fatalf("dtype mismatch: got %s want %s", actualDType, dtype.Float32)
	}

	values, err := convert.BytesToFloat32(dtype.Float32, rawBytes)

	if err != nil {
		t.Fatalf("BytesToFloat32 failed: %v", err)
	}

	return values
}
