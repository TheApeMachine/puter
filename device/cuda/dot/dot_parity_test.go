//go:build cuda

package dot

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpudot "github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestDotCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA dot product", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := parity.RandomUnaryInput(count, 0xD100+int64(count))
				right := parity.RandomUnaryInput(count, 0xD200+int64(count))
				want := []float32{cpudot.DotF32Generic(&left[0], &right[0], count)}
				partialCount := (count + 255) / 256

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer scratchTensor.Close()
				defer outputTensor.Close()

				if err := DispatchDot(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(leftTensor.Ref()),
					cudadevice.BufferRef(rightTensor.Ref()),
					cudadevice.BufferRef(scratchTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Dot: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
