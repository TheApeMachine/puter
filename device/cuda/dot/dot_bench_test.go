//go:build cuda

package dot

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkDotCUDAF32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	partialCount := (count + 255) / 256
	left := parity.RandomUnaryInput(count, 1)
	right := parity.RandomUnaryInput(count, 2)
	leftTensor := harness.UploadVector(left, dtype.Float32)
	rightTensor := harness.UploadVector(right, dtype.Float32)
	scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
	defer leftTensor.Close()
	defer rightTensor.Close()
	defer scratchTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchDot(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(leftTensor.Ref()),
			cudadevice.BufferRef(rightTensor.Ref()),
			cudadevice.BufferRef(scratchTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
