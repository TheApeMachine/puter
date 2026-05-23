//go:build cuda

package causal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpucausal "github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestCausalCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA CATE", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				treated := parity.RandomUnaryInput(count, 0xCA00+int64(count))
				control := parity.RandomUnaryInput(count, 0xCB00+int64(count))
				want := make([]float32, count)
				cpucausal.CateFloat32Native(treated, control, want)

				treatedTensor := harness.UploadVector(treated, dtype.Float32)
				controlTensor := harness.UploadVector(control, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer treatedTensor.Close()
				defer controlTensor.Close()
				defer outputTensor.Close()

				if err := DispatchCATE(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(treatedTensor.Ref()),
					cudadevice.BufferRef(controlTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch CATE: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
