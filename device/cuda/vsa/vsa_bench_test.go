//go:build cuda

package vsa

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkVSACUDABind(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	left := parity.RandomUnaryInput(count, 1)
	right := parity.RandomUnaryInput(count, 2)
	leftTensor := harness.UploadVector(left, dtype.Float32)
	rightTensor := harness.UploadVector(right, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer leftTensor.Close()
	defer rightTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchBind(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(leftTensor.Ref()),
			cudadevice.BufferRef(rightTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
