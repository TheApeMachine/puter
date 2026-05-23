//go:build cuda

package active_inference

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkActiveInferenceCUDABeliefUpdate(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	partialCount := (count + 255) / 256
	likelihood := parity.RandomUnaryInput(count, 1)
	prior := parity.RandomUnaryInput(count, 2)
	likelihoodTensor := harness.UploadVector(likelihood, dtype.Float32)
	priorTensor := harness.UploadVector(prior, dtype.Float32)
	scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer likelihoodTensor.Close()
	defer priorTensor.Close()
	defer scratchTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchBeliefUpdate(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(likelihoodTensor.Ref()),
			cudadevice.BufferRef(priorTensor.Ref()),
			cudadevice.BufferRef(scratchTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
