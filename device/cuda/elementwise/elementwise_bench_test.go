//go:build cuda

package elementwise

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkElementwiseCUDAAdd(b *testing.B) {
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
		if err := DispatchBinaryElementwiseRefs(
			harness.ContextRef(),
			outputTensor.Ref(),
			leftTensor.Ref(),
			rightTensor.Ref(),
			dtype.Float32,
			BinaryAdd,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
