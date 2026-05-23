//go:build cuda

package dropout

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkDropoutCUDAF32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	config := device.DropoutConfig{Rate: 0.25, Seed: 1}
	source := parity.RandomUnaryInput(count, 1)
	sourceTensor := harness.UploadVector(source, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchDropout(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(sourceTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			uint32(count),
			config,
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
