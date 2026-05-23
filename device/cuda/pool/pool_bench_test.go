//go:build cuda

package pool

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkPoolCUDAMaxPool2D(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	batch := uint32(1)
	channels := uint32(8)
	inHeight := uint32(64)
	inWidth := uint32(64)
	outHeight := uint32(32)
	outWidth := uint32(32)
	elementCount := int(batch * channels * inHeight * inWidth)
	input := parity.RandomUnaryInput(elementCount, 1)
	outputCount := int(batch * channels * outHeight * outWidth)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, outputCount), dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchMaxPool2D(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(inputTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			batch,
			channels,
			inHeight,
			inWidth,
			outHeight,
			outWidth,
			0,
		); err != nil {
			b.Fatal(err)
		}
	}
}
