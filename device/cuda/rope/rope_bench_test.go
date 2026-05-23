//go:build cuda

package rope

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkRoPECUDAPairs(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	halfDim := uint32(count / 2)
	input := parity.RandomUnaryInput(count, 1)
	cosValues := parity.RandomUnaryInput(int(halfDim), 2)
	sinValues := parity.RandomUnaryInput(int(halfDim), 3)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	cosTensor := harness.UploadVector(cosValues, dtype.Float32)
	sinTensor := harness.UploadVector(sinValues, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer cosTensor.Close()
	defer sinTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchRoPEPairsRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			outputTensor.Ref(),
			cosTensor.Ref(),
			sinTensor.Ref(),
			halfDim,
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
