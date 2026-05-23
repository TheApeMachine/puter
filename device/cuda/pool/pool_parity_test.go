//go:build cuda

package pool

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func maxPool2DReference(input []float32, batch, channels, inHeight, inWidth, outHeight, outWidth int) []float32 {
	output := make([]float32, batch*channels*outHeight*outWidth)
	kernelHeight := inHeight / outHeight
	kernelWidth := inWidth / outWidth

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			for outRow := 0; outRow < outHeight; outRow++ {
				for outCol := 0; outCol < outWidth; outCol++ {
					maxValue := float32(-1e30)

					for kernelRow := 0; kernelRow < kernelHeight; kernelRow++ {
						for kernelCol := 0; kernelCol < kernelWidth; kernelCol++ {
							inRow := outRow*kernelHeight + kernelRow
							inCol := outCol*kernelWidth + kernelCol
							offset := batchIndex*channels*inHeight*inWidth +
								channelIndex*inHeight*inWidth +
								inRow*inWidth + inCol
							value := input[offset]

							if value > maxValue {
								maxValue = value
							}
						}
					}

					outOffset := batchIndex*channels*outHeight*outWidth +
						channelIndex*outHeight*outWidth +
						outRow*outWidth + outCol
					output[outOffset] = maxValue
				}
			}
		}
	}

	return output
}

func TestPoolCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA max pool2d", t, func() {
		for _, count := range parity.Lengths {
			if count < 4 {
				continue
			}

			convey.Convey(fmt.Sprintf("spatial=%d", count), func() {
				batch := 1
				channels := 1
				inHeight := count
				inWidth := count
				outHeight := count / 2
				outWidth := count / 2
				elementCount := batch * channels * inHeight * inWidth
				input := parity.RandomUnaryInput(elementCount, 0xA100+int64(count))
				want := maxPool2DReference(input, batch, channels, inHeight, inWidth, outHeight, outWidth)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, len(want)), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				if err := DispatchMaxPool2D(
					parity.DeviceRef(harness.ContextRef()),
					parity.BufferRef(inputTensor.Ref()),
					parity.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(batch),
					uint32(channels),
					uint32(inHeight),
					uint32(inWidth),
					uint32(outHeight),
					uint32(outWidth),
					0,
				); err != nil {
					t.Fatalf("dispatch MaxPool2D: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
