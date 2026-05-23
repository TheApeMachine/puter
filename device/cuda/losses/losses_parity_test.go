//go:build cuda

package losses

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpulosses "github.com/theapemachine/puter/device/cpu/losses"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestLossesCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA MSE loss", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				predictions := parity.RandomUnaryInput(count, 0x2100+int64(count))
				targets := parity.RandomUnaryInput(count, 0x2200+int64(count))
				want := []float32{cpulosses.MSE(
					unsafe.Pointer(&predictions[0]),
					unsafe.Pointer(&targets[0]),
					count,
					dtype.Float32,
				)}
				partialCount := (count + 255) / 256

				predictionsTensor := harness.UploadVector(predictions, dtype.Float32)
				targetsTensor := harness.UploadVector(targets, dtype.Float32)
				scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer predictionsTensor.Close()
				defer targetsTensor.Close()
				defer scratchTensor.Close()
				defer outputTensor.Close()

				if err := DispatchPairLoss(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(predictionsTensor.Ref()),
					cudadevice.BufferRef(targetsTensor.Ref()),
					cudadevice.BufferRef(scratchTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					KernelMSE,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch MSE: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
