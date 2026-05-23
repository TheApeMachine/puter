//go:build cuda

package physics

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpuphysics "github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestPhysicsCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA grad1d", t, func() {
		invTwoDx := float32(0.5)

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := parity.RandomUnaryInput(count, 0x8100+int64(count))
				want := make([]float32, count)
				cpuphysics.Grad1DFloat32Scalar(input, want, invTwoDx)
				spacing := []float32{invTwoDx}

				inputTensor := harness.UploadVector(input, dtype.Float32)
				spacingTensor := harness.UploadVector(spacing, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer spacingTensor.Close()
				defer outputTensor.Close()

				if err := DispatchGrad1D(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(inputTensor.Ref()),
					cudadevice.BufferRef(spacingTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Grad1D: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
