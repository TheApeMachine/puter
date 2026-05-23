//go:build cuda

package sampling

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkSamplingCUDAGreedy(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	logits := parity.RandomUnaryInput(count, 1)
	logitsTensor := harness.UploadVector(logits, dtype.Float32)
	scoresTensor := harness.UploadVector(make([]float32, PaddedCount(uint32(count))), dtype.Float32)
	indicesTensor := harness.UploadVector(make([]float32, PaddedCount(uint32(count))), dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
	defer logitsTensor.Close()
	defer scoresTensor.Close()
	defer indicesTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchSampling(
			cudadevice.DeviceRef(harness.ContextRef()),
			0,
			cudadevice.BufferRef(logitsTensor.Ref()),
			cudadevice.BufferRef(scoresTensor.Ref()),
			cudadevice.BufferRef(indicesTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
			0,
		); err != nil {
			b.Fatal(err)
		}
	}
}
