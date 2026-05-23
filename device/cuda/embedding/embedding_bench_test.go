//go:build cuda

package embedding

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkEmbeddingCUDALookup(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	vocab := uint32(1024)
	hidden := uint32(64)
	count := 8192
	table := parity.RandomUnaryInput(int(vocab*hidden), 1)
	indices := make([]uint32, count)

	for index := range indices {
		indices[index] = uint32(index % int(vocab))
	}

	tableTensor := harness.UploadVector(table, dtype.Float32)
	indicesTensor := harness.UploadBytes(uint32SliceToBytes(indices))
	outputTensor := harness.UploadVector(make([]float32, count*int(hidden)), dtype.Float32)
	errorFlagTensor := harness.UploadBytes(make([]byte, 4))
	defer tableTensor.Close()
	defer indicesTensor.Close()
	defer outputTensor.Close()
	defer errorFlagTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchLookup(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(tableTensor.Ref()),
			cudadevice.BufferRef(indicesTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			cudadevice.BufferRef(errorFlagTensor.Ref()),
			dtype.Float32,
			vocab,
			hidden,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
