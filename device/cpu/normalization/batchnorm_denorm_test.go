package normalization

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestBatchNormDenormFloat32Parity(testingObject *testing.T) {
	convey.Convey("Given BatchNormDenorm float32 scalar dispatch", testingObject, func() {
		batch := 2
		channels := 3
		mean := []float32{-0.5, 0.25, 1.5}
		variance := []float32{0.01, 0.25, 1.75}
		normalization := New()

		for _, spatial := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the reference formula for N=%d", spatial), func() {
				input := randomNormalizationRow(batch*channels*spatial, 0x6D00+int64(spatial))
				got := make([]float32, len(input))
				want := batchNormDenormReference(input, mean, variance, batch, channels, spatial)

				normalization.BatchNormDenorm(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mean[0]),
					unsafe.Pointer(&variance[0]),
					unsafe.Pointer(&got[0]),
					batch,
					channels,
					spatial,
					dtype.Float32,
				)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 0)
			})
		}
	})
}

func BenchmarkBatchNormDenormFloat32(benchmark *testing.B) {
	batch := 2
	channels := 128
	spatial := 8192
	input := randomNormalizationRow(batch*channels*spatial, 0x6D10)
	mean := randomNormalizationRow(channels, 0x6D11)
	variance := positiveNormalizationRow(channels, 0x6D12)
	output := make([]float32, len(input))
	normalization := New()

	benchmark.SetBytes(int64(len(input) * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		normalization.BatchNormDenorm(
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&mean[0]),
			unsafe.Pointer(&variance[0]),
			unsafe.Pointer(&output[0]),
			batch,
			channels,
			spatial,
			dtype.Float32,
		)
	}
}

func batchNormDenormReference(
	input, mean, variance []float32,
	batch, channels, spatial int,
) []float32 {
	output := make([]float32, len(input))

	for channelIndex := 0; channelIndex < channels; channelIndex++ {
		channelMean := mean[channelIndex]
		channelStdDev := float32(math.Sqrt(float64(variance[channelIndex] + normEpsilon)))

		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial

			for spatialIndex := 0; spatialIndex < spatial; spatialIndex++ {
				output[start+spatialIndex] = input[start+spatialIndex]*channelStdDev + channelMean
			}
		}
	}

	return output
}

func positiveNormalizationRow(length int, seed int64) []float32 {
	values := randomNormalizationRow(length, seed)

	for index := range values {
		values[index] = float32(math.Abs(float64(values[index]))) + 0.01
	}

	return values
}
