//go:build cuda

package normalization

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkNormalizationCUDAInstanceNorm(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	batch := uint32(4)
	channels := uint32(8)
	spatial := uint32(1024)
	elementCount := int(batch * channels * spatial)
	input := parity.RandomUnaryInput(elementCount, 1)
	scale := parity.RandomUnaryInput(int(channels), 2)
	bias := parity.RandomUnaryInput(int(channels), 3)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	scaleTensor := harness.UploadVector(scale, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchInstanceNorm(
			parity.DeviceRef(harness.ContextRef()),
			parity.BufferRef(inputTensor.Ref()),
			parity.BufferRef(scaleTensor.Ref()),
			parity.BufferRef(biasTensor.Ref()),
			parity.BufferRef(outputTensor.Ref()),
			batch,
			channels,
			spatial,
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
