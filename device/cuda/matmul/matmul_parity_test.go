//go:build cuda

package matmul

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpumatmul "github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestMatmulCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA matmul", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d square", count), func() {
				rows := uint32(count)
				inner := uint32(count)
				cols := uint32(count)
				left := parity.RandomUnaryInput(int(rows*inner), 0x3100+int64(count))
				right := parity.RandomUnaryInput(int(inner*cols), 0x3200+int64(count))
				want := make([]float32, rows*cols)
				cpumatmul.MatmulFloat32Native(want, left, right, int(rows), int(inner), int(cols))

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, rows*cols), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer outputTensor.Close()

				if err := DispatchMatmul(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(leftTensor.Ref()),
					cudadevice.BufferRef(rightTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					rows,
					inner,
					cols,
				); err != nil {
					t.Fatalf("dispatch Matmul: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
