//go:build cuda

package reduction

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpureduction "github.com/theapemachine/puter/device/cpu/reduction"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestReductionCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA Sum reduction", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := parity.RandomUnaryInput(count, 0x1100+int64(count))
				want := []float32{cpureduction.SumFloat32Native(source)}
				partialCount := (count + 255) / 256

				sourceTensor := harness.UploadVector(source, dtype.Float32)
				scratchA := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
				scratchB := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer sourceTensor.Close()
				defer scratchA.Close()
				defer scratchB.Close()
				defer outputTensor.Close()

				if err := DispatchReductionRefs(
					harness.ContextRef(),
					sourceTensor.Ref(),
					scratchA.Ref(),
					scratchB.Ref(),
					outputTensor.Ref(),
					dtype.Float32,
					KernelSum,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Sum: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
