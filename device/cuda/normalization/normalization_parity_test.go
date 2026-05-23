//go:build cuda

package normalization

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func instanceNormReference(input, scale, bias []float32, batch, channels, spatial int) []float32 {
	output := make([]float32, len(input))

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			offset := batchIndex*channels*spatial + channelIndex*spatial
			channel := input[offset : offset+spatial]
			var mean float32

			for _, value := range channel {
				mean += value
			}

			mean /= float32(spatial)

			var variance float32

			for _, value := range channel {
				diff := value - mean
				variance += diff * diff
			}

			invStdDev := float32(1.0) / float32(sqrtFloat32(variance/float32(spatial)+1e-5))

			for spatialIndex := 0; spatialIndex < spatial; spatialIndex++ {
				normalized := (channel[spatialIndex] - mean) * invStdDev
				output[offset+spatialIndex] = normalized*scale[channelIndex] + bias[channelIndex]
			}
		}
	}

	return output
}

func sqrtFloat32(value float32) float32 {
	// Newton iteration for test reference only.
	guess := value

	if guess <= 0 {
		return 0
	}

	for range 8 {
		guess = 0.5 * (guess + value/guess)
	}

	return guess
}

func TestNormalizationCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA instance norm", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("spatial=%d", count), func() {
				batch := uint32(1)
				channels := uint32(1)
				spatial := uint32(count)
				elementCount := int(batch * channels * spatial)
				input := parity.RandomUnaryInput(elementCount, 0x3100+int64(count))
				scale := parity.RandomUnaryInput(int(channels), 0x3200+int64(count))
				bias := parity.RandomUnaryInput(int(channels), 0x3300+int64(count))
				want := instanceNormReference(input, scale, bias, int(batch), int(channels), int(spatial))

				inputTensor := harness.UploadVector(input, dtype.Float32)
				scaleTensor := harness.UploadVector(scale, dtype.Float32)
				biasTensor := harness.UploadVector(bias, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
				defer inputTensor.Close()
				defer scaleTensor.Close()
				defer biasTensor.Close()
				defer outputTensor.Close()

				if err := DispatchInstanceNorm(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(inputTensor.Ref()),
					cudadevice.BufferRef(scaleTensor.Ref()),
					cudadevice.BufferRef(biasTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					batch,
					channels,
					spatial,
					dtype.Float32,
				); err != nil {
					t.Fatalf("dispatch InstanceNorm: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 3)
			})
		}
	})
}
