//go:build cuda

package physics

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkPhysicsCUDAGrad1D(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	input := parity.RandomUnaryInput(count, 1)
	spacing := []float32{0.5}
	inputTensor := harness.UploadVector(input, dtype.Float32)
	spacingTensor := harness.UploadVector(spacing, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer spacingTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchGrad1D(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(inputTensor.Ref()),
			parity.BufferRef(spacingTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
