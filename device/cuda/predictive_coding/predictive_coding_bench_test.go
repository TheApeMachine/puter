//go:build cuda

package predictive_coding

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkPredictiveCodingCUDAPrediction(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	outDim := 64
	inDim := 64
	weights := parity.RandomUnaryInput(outDim*inDim, 1)
	state := parity.RandomUnaryInput(inDim, 2)
	weightsTensor := harness.UploadVector(weights, dtype.Float32)
	stateTensor := harness.UploadVector(state, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, outDim), dtype.Float32)
	defer weightsTensor.Close()
	defer stateTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchPrediction(
			cudadevice.DeviceRef(harness.ContextRef()),
			cudadevice.BufferRef(weightsTensor.Ref()),
			cudadevice.BufferRef(stateTensor.Ref()),
			cudadevice.BufferRef(outputTensor.Ref()),
			dtype.Float32,
			uint32(outDim),
			uint32(inDim),
		); err != nil {
			b.Fatal(err)
		}
	}
}
