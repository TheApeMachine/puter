//go:build darwin && cgo

package convolution

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestConv2DMetalFloat32Parity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal Conv2D float32 kernels", testingObject, func() {
		for _, width := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match pointwise 1x1 convolution for N=%d", width), func() {
				input := randomConvVector(width, 0x8100+int64(width))
				weight := []float32{0.75}
				bias := []float32{-0.125}
				want := make([]float32, len(input))

				for index := range input {
					want[index] = input[index]*weight[0] + bias[0]
				}

				inputTensor := harness.UploadVector(input, dtype.Float32)
				weightTensor := harness.UploadVector(weight, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, len(input)), dtype.Float32)
				defer inputTensor.Close()
				defer weightTensor.Close()
				defer biasTensor.Close()
				defer outputTensor.Close()

				if err := DispatchConv2DRefs(
					harness.ContextRef(),
					inputTensor.Ref(),
					weightTensor.Ref(),
					biasTensor.Ref(),
					outputTensor.Ref(),
					dtype.Float32,
					1,
					1,
					1,
					uint32(width),
					1,
					1,
					1,
					1,
					uint32(width),
					1,
					1,
					0,
					0,
					1,
					1,
				); err != nil {
					testingObject.Fatalf("dispatch Conv2D: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 1)
			})
		}
	})
}

func TestConv2DMetalStridePaddingDilationParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given a strided padded dilated Conv2D on Metal", testingObject, func() {
		input := []float32{
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15,
			16, 17, 18, 19, 20,
			21, 22, 23, 24, 25,
		}
		weight := []float32{
			0.5, -0.25,
			0.75, 0.125,
		}
		bias := []float32{0.1}
		want := conv2DReference(input, weight, bias, 1, 1, 5, 5, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1)

		inputTensor := harness.UploadVector(input, dtype.Float32)
		weightTensor := harness.UploadVector(weight, dtype.Float32)
		biasTensor := harness.UploadVector(bias, dtype.Float32)
		outputTensor := harness.UploadVector(make([]float32, len(want)), dtype.Float32)
		defer inputTensor.Close()
		defer weightTensor.Close()
		defer biasTensor.Close()
		defer outputTensor.Close()

		if err := DispatchConv2DRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			weightTensor.Ref(),
			biasTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			1,
			1,
			5,
			5,
			1,
			2,
			2,
			3,
			3,
			2,
			2,
			1,
			1,
			1,
			1,
		); err != nil {
			testingObject.Fatalf("dispatch Conv2D: %v", err)
		}

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 1)
	})
}

func BenchmarkConv2DMetalFloat32(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	batch := 1
	inChannels := 32
	outChannels := 32
	height := 64
	width := 64
	kernel := 3
	elementCount := batch * outChannels * height * width
	input := randomConvVector(batch*inChannels*height*width, 0x8200)
	weight := randomConvVector(outChannels*inChannels*kernel*kernel, 0x8201)
	bias := randomConvVector(outChannels, 0x8202)

	inputTensor := harness.UploadVector(input, dtype.Float32)
	weightTensor := harness.UploadVector(weight, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer weightTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	benchmark.SetBytes(int64(elementCount * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := DispatchConv2DRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			weightTensor.Ref(),
			biasTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(batch),
			uint32(inChannels),
			uint32(height),
			uint32(width),
			uint32(outChannels),
			uint32(kernel),
			uint32(kernel),
			uint32(height),
			uint32(width),
			1,
			1,
			1,
			1,
			1,
			1,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func randomConvVector(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = rng.Float32()*2.0 - 1.0
	}

	return values
}

func conv2DReference(
	input, weight, bias []float32,
	batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth, strideHeight, strideWidth, paddingHeight, paddingWidth,
	dilationHeight, dilationWidth int,
) []float32 {
	output := make([]float32, batch*outChannels*outHeight*outWidth)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outChannel := 0; outChannel < outChannels; outChannel++ {
			for outRow := 0; outRow < outHeight; outRow++ {
				for outCol := 0; outCol < outWidth; outCol++ {
					outputIndex := ((batchIndex*outChannels+outChannel)*outHeight+outRow)*outWidth + outCol
					output[outputIndex] = conv2DReferenceValue(
						input,
						weight,
						bias[outChannel],
						batchIndex,
						outChannel,
						outRow,
						outCol,
						inChannels,
						inHeight,
						inWidth,
						kernelHeight,
						kernelWidth,
						strideHeight,
						strideWidth,
						paddingHeight,
						paddingWidth,
						dilationHeight,
						dilationWidth,
					)
				}
			}
		}
	}

	return output
}

func conv2DReferenceValue(
	input, weight []float32,
	biasValue float32,
	batchIndex, outChannel, outRow, outCol,
	inChannels, inHeight, inWidth, kernelHeight, kernelWidth,
	strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth int,
) float32 {
	sum := biasValue
	baseRow := outRow*strideHeight - paddingHeight
	baseCol := outCol*strideWidth - paddingWidth

	for inChannel := 0; inChannel < inChannels; inChannel++ {
		for kernelRow := 0; kernelRow < kernelHeight; kernelRow++ {
			inputRow := baseRow + kernelRow*dilationHeight

			if inputRow < 0 || inputRow >= inHeight {
				continue
			}

			for kernelCol := 0; kernelCol < kernelWidth; kernelCol++ {
				inputCol := baseCol + kernelCol*dilationWidth

				if inputCol < 0 || inputCol >= inWidth {
					continue
				}

				inputIndex := ((batchIndex*inChannels+inChannel)*inHeight+inputRow)*inWidth + inputCol
				weightIndex := ((outChannel*inChannels+inChannel)*kernelHeight+kernelRow)*kernelWidth + kernelCol
				sum += input[inputIndex] * weight[weightIndex]
			}
		}
	}

	return sum
}
