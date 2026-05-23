//go:build cuda

package activation

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkActivationCUDAReLU(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := parity.RandomUnaryInput(count, 1)
	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchStandardUnary(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(destinationTensor.Ref()),
			cudadevice.BufferRef(sourceTensor.Ref()),
			dtype.Float32,
			StandardReLU,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
