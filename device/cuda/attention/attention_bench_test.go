//go:build cuda

package attention

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkAttentionCUDAApplyMask(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	input := parity.RandomUnaryInput(count, 1)
	mask := parity.RandomUnaryInput(count, 2)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	maskTensor := harness.UploadVector(mask, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer maskTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchApplyMask(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(inputTensor.Ref()),
			parity.BufferRef(maskTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			uint32(count),
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
