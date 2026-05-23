//go:build cuda

package quant

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkQuantCUDAInt8(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := parity.RandomUnaryInput(count, 1)
	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := harness.UploadBytes(make([]byte, count))
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchQuant(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(sourceTensor.Ref()),
			parity.BufferRef(destinationTensor.Ref()),
			0.0875,
			-13,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
