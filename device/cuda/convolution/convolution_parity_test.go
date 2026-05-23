//go:build cuda

package convolution

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func conv1DReference(
	input, weight, bias []float32,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) []float32 {
	output := make([]float32, batch*outChannels*outLength)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outChannel := 0; outChannel < outChannels; outChannel++ {
			for outIndex := 0; outIndex < outLength; outIndex++ {
				var sum float32

				for inChannel := 0; inChannel < inChannels; inChannel++ {
					for kernelIndex := 0; kernelIndex < kernelLength; kernelIndex++ {
						inIndex := outIndex + kernelIndex
						inputOffset := batchIndex*inChannels*inLength + inChannel*inLength + inIndex
						weightOffset := outChannel*inChannels*kernelLength + inChannel*kernelLength + kernelIndex
						sum += input[inputOffset] * weight[weightOffset]
					}
				}

				outputOffset := batchIndex*outChannels*outLength + outChannel*outLength + outIndex
				output[outputOffset] = sum + bias[outChannel]
			}
		}
	}

	return output
}

func TestConvolutionCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA conv1d", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				batch := uint32(1)
				inChannels := uint32(1)
				outChannels := uint32(1)
				kernelLength := uint32(3)
				inLength := uint32(count)
				outLength := inLength - kernelLength + 1

				if outLength == 0 {
					return
				}

				input := parity.RandomUnaryInput(int(batch*inChannels*inLength), 0xC100+int64(count))
				weight := parity.RandomUnaryInput(int(outChannels*inChannels*kernelLength), 0xC200+int64(count))
				bias := parity.RandomUnaryInput(int(outChannels), 0xC300+int64(count))
				want := conv1DReference(
					input, weight, bias,
					int(batch), int(inChannels), int(inLength), int(outChannels), int(kernelLength), int(outLength),
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				weightTensor := harness.UploadVector(weight, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, len(want)), dtype.Float32)
				defer inputTensor.Close()
				defer weightTensor.Close()
				defer biasTensor.Close()
				defer outputTensor.Close()

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
					t.Fatalf("dispatch Conv1D: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
