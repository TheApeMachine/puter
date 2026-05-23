//go:build cuda

package causal

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkCausalCUDACATE(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	treated := parity.RandomUnaryInput(count, 1)
	control := parity.RandomUnaryInput(count, 2)
	treatedTensor := harness.UploadVector(treated, dtype.Float32)
	controlTensor := harness.UploadVector(control, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer treatedTensor.Close()
	defer controlTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchCATE(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(treatedTensor.Ref()),
			parity.BufferRef(controlTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
