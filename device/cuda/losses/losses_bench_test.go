//go:build cuda

package losses

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkLossesCUDAMSE(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	partialCount := (count + 255) / 256
	predictions := parity.RandomUnaryInput(count, 1)
	targets := parity.RandomUnaryInput(count, 2)
	predictionsTensor := harness.UploadVector(predictions, dtype.Float32)
	targetsTensor := harness.UploadVector(targets, dtype.Float32)
	scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
	defer predictionsTensor.Close()
	defer targetsTensor.Close()
	defer scratchTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchPairLossRefs(
			harness.ContextRef(),
			predictionsTensor.Ref(),
			targetsTensor.Ref(),
			scratchTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			KernelMSE,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
