//go:build cuda

package convolution

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func BenchmarkConvolutionCUDAConv1D(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	batch := uint32(1)
	inChannels := uint32(4)
	outChannels := uint32(8)
	kernelLength := uint32(3)
	inLength := uint32(8192)
	outLength := inLength - kernelLength + 1
	input := parity.RandomUnaryInput(int(batch*inChannels*inLength), 1)
	weight := parity.RandomUnaryInput(int(outChannels*inChannels*kernelLength), 2)
	bias := parity.RandomUnaryInput(int(outChannels), 3)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	weightTensor := harness.UploadVector(weight, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, batch*outChannels*outLength), dtype.Float32)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchConv1DRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			weightTensor.Ref(),
			biasTensor.Ref(),
			outputTensor.Ref(),
			batch,
			inChannels,
			inLength,
			outChannels,
			kernelLength,
			outLength,
			dtype.Float32,
		); err != nil {
			b.Fatal(err)
		}
	}
}
