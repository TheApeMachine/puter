//go:build cuda

package reduction

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkReductionCUDASum(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	partialCount := (count + 255) / 256
	source := parity.RandomUnaryInput(count, 1)
	sourceTensor := harness.UploadVector(source, dtype.Float32)
	scratchA := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
	scratchB := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
	defer sourceTensor.Close()
	defer scratchA.Close()
	defer scratchB.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchReduction(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(sourceTensor.Ref()),
			parity.BufferRef(scratchA.Ref()),
			parity.BufferRef(scratchB.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			KernelSum,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
