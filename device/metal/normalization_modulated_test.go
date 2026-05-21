package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

func TestMetalModulatedLayerNorm(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given input rows and shift/scale modulation", t, func() {
		inputShape := mustShapeForTest(t, []int{1, 2, 3})
		modulationShape := mustShapeForTest(t, []int{1, 9})
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
			convert.Float32ToBytes([]float32{1, -1, 0.25, 0.5, 1, -0.5, 0, 0, 0}),
		)
		output := emptyTensorForTest(t, backend, outputShape, dtype.Float32)

		defer closeBenchmarkTensors(input, modulation, output)

		convey.Convey("It should normalize each row and apply broadcast modulation", func() {
			err := runMetalModulatedLayerNorm(input, modulation, output, 0)

			convey.So(err, convey.ShouldBeNil)
			assertFloat32WithinULP(
				t,
				mustDownloadFloat32ForTest(t, output),
				expectedModulatedLayerNormValues(),
				32,
			)
		})
	})
}

func TestMetalGatedResidual(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given residual rows, branch rows, and gate modulation", t, func() {
		inputShape := mustShapeForTest(t, []int{1, 2, 3})
		modulationShape := mustShapeForTest(t, []int{1, 9})
		outputShape := mustShapeForTest(t, []int{1, 2, 3})
		residual := uploadDTypeTensorForTest(
			t,
			backend,
			inputShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{1, 2, 3, 4, 5, 6}),
		)
		branch := uploadDTypeTensorForTest(
			t,
			backend,
			inputShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{10, 20, 30, 40, 50, 60}),
		)
		modulation := uploadDTypeTensorForTest(
			t,
			backend,
			modulationShape,
			dtype.Float32,
			convert.Float32ToBytes([]float32{0, 0, 0, 0, 0, 0, 0.5, 1, -0.5}),
		)
		output := emptyTensorForTest(t, backend, outputShape, dtype.Float32)

		defer closeBenchmarkTensors(residual, branch, modulation, output)

		convey.Convey("It should broadcast the gate over sequence rows", func() {
			err := runMetalGatedResidual(residual, branch, modulation, output, 0)

			convey.So(err, convey.ShouldBeNil)
			assertFloat32WithinULP(
				t,
				mustDownloadFloat32ForTest(t, output),
				[]float32{6, 22, -12, 24, 55, -24},
				32,
			)
		})
	})
}

func expectedModulatedLayerNormValues() []float32 {
	input := []float32{1, 2, 3, 2, 4, 6}
	shift := []float32{1, -1, 0.25}
	scale := []float32{0.5, 1, -0.5}
	output := make([]float32, len(input))

	for row := range 2 {
		rowOffset := row * 3
		mean := (input[rowOffset] + input[rowOffset+1] + input[rowOffset+2]) / 3
		variance := float32(0)

		for col := range 3 {
			delta := input[rowOffset+col] - mean
			variance += delta * delta
		}

		invStdDev := 1 / float32(math.Sqrt(float64(variance/3+layerNormEpsilonMetalForTest)))

		for col := range 3 {
			normalized := (input[rowOffset+col] - mean) * invStdDev
			output[rowOffset+col] = normalized*(1+scale[col]) + shift[col]
		}
	}

	return output
}
