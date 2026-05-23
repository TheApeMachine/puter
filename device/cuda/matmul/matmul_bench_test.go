//go:build cuda

package matmul

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkMatmulCUDAF32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	rows := uint32(64)
	inner := uint32(64)
	cols := uint32(64)
	left := parity.RandomUnaryInput(int(rows*inner), 1)
	right := parity.RandomUnaryInput(int(inner*cols), 2)
	leftTensor := harness.UploadVector(left, dtype.Float32)
	rightTensor := harness.UploadVector(right, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, rows*cols), dtype.Float32)
	defer leftTensor.Close()
	defer rightTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchMatmul(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(leftTensor.Ref()),
			parity.BufferRef(rightTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			rows,
			inner,
			cols,
		); err != nil {
			b.Fatal(err)
		}
	}
}
