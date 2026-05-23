//go:build cuda

package hawkes

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkHawkesCUDAIntensity(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	eventTimes := parity.RandomUnaryInput(count, 1)
	queryTimes := parity.RandomUnaryInput(count, 2)
	eventsTensor := harness.UploadVector(eventTimes, dtype.Float32)
	queryTensor := harness.UploadVector(queryTimes, dtype.Float32)
	baselineTensor := harness.UploadVector([]float32{0.1}, dtype.Float32)
	alphaTensor := harness.UploadVector([]float32{0.5}, dtype.Float32)
	betaTensor := harness.UploadVector([]float32{1.0}, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer eventsTensor.Close()
	defer queryTensor.Close()
	defer baselineTensor.Close()
	defer alphaTensor.Close()
	defer betaTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchHawkesIntensity(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(eventsTensor.Ref()),
			cudadevice.BufferRef(queryTensor.Ref()),
			cudadevice.BufferRef(baselineTensor.Ref()),
			cudadevice.BufferRef(alphaTensor.Ref()),
			cudadevice.BufferRef(betaTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			uint32(count),
			uint32(count),
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
