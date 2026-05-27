package embedding

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const timestepEmbeddingMaxULP = 0

func TestTimestepEmbedding(testingObject *testing.T) {
	convey.Convey("Given TimestepEmbedding float32", testingObject, func() {
		config := device.TimestepEmbeddingConfig{
			MaxPeriod:          10000,
			DownscaleFreqShift: 0,
			TimestepDivisor:    1000,
			FlipSinToCos:       true,
		}

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar reference for count=%d", count), func() {
				const dim = 256

				timesteps := make([]float32, count)

				for index := range timesteps {
					timesteps[index] = float32(index+1) * 125
				}

				want := referenceTimestepEmbedding(config, timesteps, dim)
				got := make([]float32, count*dim)

				Default.TimestepEmbedding(
					config,
					unsafe.Pointer(&timesteps[0]),
					unsafe.Pointer(&got[0]),
					count,
					dim,
					dtype.Float32,
				)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, timestepEmbeddingMaxULP)
			})
		}
	})
}

func BenchmarkTimestepEmbedding(benchmark *testing.B) {
	config := device.TimestepEmbeddingConfig{
		MaxPeriod:          10000,
		DownscaleFreqShift: 0,
		TimestepDivisor:    1000,
		FlipSinToCos:       true,
	}
	timesteps := make([]float32, 8192)
	output := make([]float32, 8192*256)

	for index := range timesteps {
		timesteps[index] = float32(index+1) * 125
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		Default.TimestepEmbedding(
			config,
			unsafe.Pointer(&timesteps[0]),
			unsafe.Pointer(&output[0]),
			8192,
			256,
			dtype.Float32,
		)
	}
}

func referenceTimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps []float32,
	dim int,
) []float32 {
	output := make([]float32, len(timesteps)*dim)

	for rowIndex, timestep := range timesteps {
		for dimIndex := range dim {
			output[rowIndex*dim+dimIndex] = referenceTimestepEmbeddingValue(config, timestep, dim, dimIndex)
		}
	}

	return output
}

func referenceTimestepEmbeddingValue(
	config device.TimestepEmbeddingConfig,
	timestep float32,
	dim int,
	dimIndex int,
) float32 {
	halfDim := dim / 2

	if halfDim == 0 || dimIndex >= halfDim*2 {
		return 0
	}

	firstHalf := dimIndex < halfDim
	frequencyIndex := dimIndex

	if !firstHalf {
		frequencyIndex -= halfDim
	}

	denominator := float32(halfDim) - config.DownscaleFreqShift
	exponent := -float32(math.Log(float64(config.MaxPeriod))) * float32(frequencyIndex) / denominator
	angle := float64((timestep / config.TimestepDivisor) * float32(math.Exp(float64(exponent))))
	sinValue := float32(math.Sin(angle))
	cosValue := float32(math.Cos(angle))

	if config.FlipSinToCos {
		if firstHalf {
			return cosValue
		}

		return sinValue
	}

	if firstHalf {
		return sinValue
	}

	return cosValue
}
