//go:build cuda

package elementwise

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuelementwise "github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestElementwiseCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA binary Add kernels", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := parity.RandomUnaryInput(count, 0xE100+int64(count))
				right := parity.RandomUnaryInput(count, 0xE200+int64(count))
				want := make([]float32, count)
				cpuelementwise.AddF32Generic(&want[0], &left[0], &right[0], count)

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer outputTensor.Close()

				if err := DispatchBinaryElementwiseRefs(
					harness.ContextRef(),
					outputTensor.Ref(),
					leftTensor.Ref(),
					rightTensor.Ref(),
					dtype.Float32,
					BinaryAdd,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Add: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
