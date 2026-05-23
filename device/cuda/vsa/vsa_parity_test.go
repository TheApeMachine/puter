//go:build cuda

package vsa

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuvsa "github.com/theapemachine/puter/device/cpu/vsa"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestVSACUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA VSA bind", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				left := parity.RandomUnaryInput(count, 0x7100+int64(count))
				right := parity.RandomUnaryInput(count, 0x7200+int64(count))
				want := make([]float32, count)
				cpuvsa.VsaBindFloat32Scalar(want, left, right)

				leftTensor := harness.UploadVector(left, dtype.Float32)
				rightTensor := harness.UploadVector(right, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer leftTensor.Close()
				defer rightTensor.Close()
				defer outputTensor.Close()

				if err := DispatchBind(
					parity.DeviceRef(harness.ContextRef()),
					parity.BufferRef(leftTensor.Ref()),
					parity.BufferRef(rightTensor.Ref()),
					parity.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Bind: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
