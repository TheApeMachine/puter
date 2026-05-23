//go:build cuda

package attention

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func applyMaskReference(input, mask []float32) []float32 {
	output := make([]float32, len(input))

	for index := range input {
		output[index] = input[index] + mask[index]
	}

	return output
}

func TestAttentionCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA apply mask", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := parity.RandomUnaryInput(count, 0xA100+int64(count))
				mask := parity.RandomUnaryInput(count, 0xA200+int64(count))
				want := applyMaskReference(input, mask)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				maskTensor := harness.UploadVector(mask, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer maskTensor.Close()
				defer outputTensor.Close()

				if err := DispatchApplyMask(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(inputTensor.Ref()),
					cudadevice.BufferRef(maskTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					uint32(count),
					dtype.Float32,
				); err != nil {
					t.Fatalf("dispatch ApplyMask: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 1)
			})
		}
	})
}
