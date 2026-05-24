//go:build xla

package predictive_coding_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpupredictive "github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referencePredictiveCoding = cpupredictive.New()

func TestPredictiveCodingXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA Prediction", t, func() {
		outDim := 64
		inDim := 32

		weights := xlaparity.RandomUnaryInput(outDim*inDim, 0x9100)
		representation := xlaparity.RandomUnaryInput(inDim, 0x9200)
		want := make([]float32, outDim)
		referencePredictiveCoding.Prediction(
			unsafe.Pointer(&weights[0]),
			unsafe.Pointer(&representation[0]),
			unsafe.Pointer(&want[0]),
			outDim, inDim,
			dtype.Float32,
		)

		weightsTensor := harness.UploadMatrix(weights, outDim, inDim, dtype.Float32)
		representationTensor := harness.UploadVector(representation, dtype.Float32)
		outputTensor := harness.UploadVector(make([]float32, outDim), dtype.Float32)
		defer weightsTensor.Close()
		defer representationTensor.Close()
		defer outputTensor.Close()

		harness.Backend().Prediction(
			xla.ResidentPointer(weightsTensor),
			xla.ResidentPointer(representationTensor),
			xla.ResidentPointer(outputTensor),
			outDim, inDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA UpdateRepresentation", t, func() {
		outDim := 7
		inDim := 64
		config := device.PredictiveCodingConfig{LearningRate: 0.01}

		weights := xlaparity.RandomUnaryInput(outDim*inDim, 0x9300)
		representation := xlaparity.RandomUnaryInput(inDim, 0x9400)
		predictionError := xlaparity.RandomUnaryInput(outDim, 0x9500)
		want := make([]float32, inDim)
		referencePredictiveCoding.UpdateRepresentation(
			config,
			unsafe.Pointer(&weights[0]),
			unsafe.Pointer(&representation[0]),
			unsafe.Pointer(&predictionError[0]),
			unsafe.Pointer(&want[0]),
			outDim, inDim,
			dtype.Float32,
		)

		weightsTensor := harness.UploadMatrix(weights, outDim, inDim, dtype.Float32)
		representationTensor := harness.UploadVector(representation, dtype.Float32)
		errorTensor := harness.UploadVector(predictionError, dtype.Float32)
		outputTensor := harness.UploadVector(make([]float32, inDim), dtype.Float32)
		defer weightsTensor.Close()
		defer representationTensor.Close()
		defer errorTensor.Close()
		defer outputTensor.Close()

		harness.Backend().UpdateRepresentation(
			config,
			xla.ResidentPointer(weightsTensor),
			xla.ResidentPointer(representationTensor),
			xla.ResidentPointer(errorTensor),
			xla.ResidentPointer(outputTensor),
			outDim, inDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA PredictionError", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				observed := xlaparity.RandomUnaryInput(count, 0x9800+int64(count))
				predicted := xlaparity.RandomUnaryInput(count, 0x9900+int64(count))
				want := make([]float32, count)
				referencePredictiveCoding.PredictionError(
					unsafe.Pointer(&observed[0]),
					unsafe.Pointer(&predicted[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				observedTensor := harness.UploadVector(observed, dtype.Float32)
				predictedTensor := harness.UploadVector(predicted, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer observedTensor.Close()
				defer predictedTensor.Close()
				defer outputTensor.Close()

				harness.Backend().PredictionError(
					xla.ResidentPointer(observedTensor),
					xla.ResidentPointer(predictedTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})

	convey.Convey("Given XLA UpdateWeights", t, func() {
		outDim := 7
		inDim := 64
		config := device.PredictiveCodingConfig{LearningRate: 0.01}

		weights := xlaparity.RandomUnaryInput(outDim*inDim, 0x9a00)
		representation := xlaparity.RandomUnaryInput(inDim, 0x9b00)
		predictionError := xlaparity.RandomUnaryInput(outDim, 0x9c00)
		want := make([]float32, outDim*inDim)
		referencePredictiveCoding.UpdateWeights(
			config,
			unsafe.Pointer(&weights[0]),
			unsafe.Pointer(&representation[0]),
			unsafe.Pointer(&predictionError[0]),
			unsafe.Pointer(&want[0]),
			outDim, inDim,
			dtype.Float32,
		)

		weightsTensor := harness.UploadMatrix(weights, outDim, inDim, dtype.Float32)
		representationTensor := harness.UploadVector(representation, dtype.Float32)
		errorTensor := harness.UploadVector(predictionError, dtype.Float32)
		outputTensor := harness.UploadMatrix(make([]float32, outDim*inDim), outDim, inDim, dtype.Float32)
		defer weightsTensor.Close()
		defer representationTensor.Close()
		defer errorTensor.Close()
		defer outputTensor.Close()

		harness.Backend().UpdateWeights(
			config,
			xla.ResidentPointer(weightsTensor),
			xla.ResidentPointer(representationTensor),
			xla.ResidentPointer(errorTensor),
			xla.ResidentPointer(outputTensor),
			outDim, inDim,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})
}

func BenchmarkPredictionXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	outDim := 128
	inDim := 128
	weights := xlaparity.RandomUnaryInput(outDim*inDim, 0x9600)
	representation := xlaparity.RandomUnaryInput(inDim, 0x9700)
	weightsTensor := harness.UploadMatrix(weights, outDim, inDim, dtype.Float32)
	representationTensor := harness.UploadVector(representation, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, outDim), dtype.Float32)
	defer weightsTensor.Close()
	defer representationTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Prediction(
			xla.ResidentPointer(weightsTensor),
			xla.ResidentPointer(representationTensor),
			xla.ResidentPointer(outputTensor),
			outDim, inDim,
			dtype.Float32,
		)
	}
}
