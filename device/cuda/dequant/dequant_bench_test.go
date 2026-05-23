//go:build cuda

package dequant

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkDequantCUDAInt8(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := randomInt8Slice(count, 1)
	sourceTensor := harness.UploadBytes(int8ToBytes(source))
	destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchDequant(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(sourceTensor.Ref()),
			parity.BufferRef(destinationTensor.Ref()),
			dtype.Float32,
			0.0875,
			-13,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
