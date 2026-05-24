//go:build xla

package active_inference_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuactive "github.com/theapemachine/puter/device/cpu/active_inference"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceActiveInference = cpuactive.New()

func TestActiveInferenceXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA BeliefUpdate", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				likelihood := xlaparity.RandomUnaryInput(count, 0x8100+int64(count))
				prior := xlaparity.RandomUnaryInput(count, 0x8200+int64(count))
				want := make([]float32, count)
				referenceActiveInference.BeliefUpdate(
					unsafe.Pointer(&likelihood[0]),
					unsafe.Pointer(&prior[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				likelihoodTensor := harness.UploadVector(likelihood, dtype.Float32)
				priorTensor := harness.UploadVector(prior, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer likelihoodTensor.Close()
				defer priorTensor.Close()
				defer outputTensor.Close()

				harness.Backend().BeliefUpdate(
					xla.ResidentPointer(likelihoodTensor),
					xla.ResidentPointer(priorTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA FreeEnergy", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				likelihood := xlaparity.RandomUnaryInput(count, 0x8500+int64(count))
				posterior := xlaparity.RandomUnaryInput(count, 0x8600+int64(count))
				prior := xlaparity.RandomUnaryInput(count, 0x8700+int64(count))
				want := make([]float32, count)
				referenceActiveInference.FreeEnergy(
					unsafe.Pointer(&likelihood[0]),
					unsafe.Pointer(&posterior[0]),
					unsafe.Pointer(&prior[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				likelihoodTensor := harness.UploadVector(likelihood, dtype.Float32)
				posteriorTensor := harness.UploadVector(posterior, dtype.Float32)
				priorTensor := harness.UploadVector(prior, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer likelihoodTensor.Close()
				defer posteriorTensor.Close()
				defer priorTensor.Close()
				defer outputTensor.Close()

				harness.Backend().FreeEnergy(
					xla.ResidentPointer(likelihoodTensor),
					xla.ResidentPointer(posteriorTensor),
					xla.ResidentPointer(priorTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA PrecisionWeight", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				errorsIn := xlaparity.RandomUnaryInput(count, 0x8800+int64(count))
				precision := xlaparity.RandomUnaryInput(count, 0x8900+int64(count))
				want := make([]float32, count)
				referenceActiveInference.PrecisionWeight(
					unsafe.Pointer(&errorsIn[0]),
					unsafe.Pointer(&precision[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				errorsTensor := harness.UploadVector(errorsIn, dtype.Float32)
				precisionTensor := harness.UploadVector(precision, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer errorsTensor.Close()
				defer precisionTensor.Close()
				defer outputTensor.Close()

				harness.Backend().PrecisionWeight(
					xla.ResidentPointer(errorsTensor),
					xla.ResidentPointer(precisionTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA ExpectedFreeEnergy", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				predictedObs := xlaparity.RandomUnaryInput(count, 0x8a00+int64(count))
				preferredObs := xlaparity.RandomUnaryInput(count, 0x8b00+int64(count))
				predictedState := xlaparity.RandomUnaryInput(count, 0x8c00+int64(count))
				want := make([]float32, 1)
				referenceActiveInference.ExpectedFreeEnergy(
					unsafe.Pointer(&predictedObs[0]),
					unsafe.Pointer(&preferredObs[0]),
					unsafe.Pointer(&predictedState[0]),
					unsafe.Pointer(&want[0]),
					count, count,
					dtype.Float32,
				)

				predictedObsTensor := harness.UploadVector(predictedObs, dtype.Float32)
				preferredObsTensor := harness.UploadVector(preferredObs, dtype.Float32)
				predictedStateTensor := harness.UploadVector(predictedState, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer predictedObsTensor.Close()
				defer preferredObsTensor.Close()
				defer predictedStateTensor.Close()
				defer outputTensor.Close()

				harness.Backend().ExpectedFreeEnergy(
					xla.ResidentPointer(predictedObsTensor),
					xla.ResidentPointer(preferredObsTensor),
					xla.ResidentPointer(predictedStateTensor),
					xla.ResidentPointer(outputTensor),
					count, count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})
}

func BenchmarkExpectedFreeEnergyXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	predictedObs := xlaparity.RandomUnaryInput(count, 0x8d00)
	preferredObs := xlaparity.RandomUnaryInput(count, 0x8e00)
	predictedState := xlaparity.RandomUnaryInput(count, 0x8f00)
	predictedObsTensor := harness.UploadVector(predictedObs, dtype.Float32)
	preferredObsTensor := harness.UploadVector(preferredObs, dtype.Float32)
	predictedStateTensor := harness.UploadVector(predictedState, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
	defer predictedObsTensor.Close()
	defer preferredObsTensor.Close()
	defer predictedStateTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().ExpectedFreeEnergy(
			xla.ResidentPointer(predictedObsTensor),
			xla.ResidentPointer(preferredObsTensor),
			xla.ResidentPointer(predictedStateTensor),
			xla.ResidentPointer(outputTensor),
			count, count,
			dtype.Float32,
		)
	}
}

func BenchmarkBeliefUpdateXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	likelihood := xlaparity.RandomUnaryInput(count, 0x8300)
	prior := xlaparity.RandomUnaryInput(count, 0x8400)
	likelihoodTensor := harness.UploadVector(likelihood, dtype.Float32)
	priorTensor := harness.UploadVector(prior, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer likelihoodTensor.Close()
	defer priorTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().BeliefUpdate(
			xla.ResidentPointer(likelihoodTensor),
			xla.ResidentPointer(priorTensor),
			xla.ResidentPointer(outputTensor),
			count,
			dtype.Float32,
		)
	}
}
