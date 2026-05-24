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
