//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

const timestepEmbeddingMetalMaxULP = 64

func TestTimestepEmbeddingMetalParity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given timestep embedding inputs on Metal", testingObject, func() {
		config := device.TimestepEmbeddingConfig{
			MaxPeriod:          10000,
			DownscaleFreqShift: 0,
			TimestepDivisor:    1000,
			FlipSinToCos:       true,
		}

		for _, count := range cpuparity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CPU for count=%d", count), func() {
				const dim = 256

				timestepValues := diffusionTimestepValues(count)

				want := referenceTimestepEmbedding(config, timestepValues, dim)
				timesteps := uploadRoPETensor(testingObject, backend, timestepValues)
				defer timesteps.Close()
				output := uploadRoPETensor(testingObject, backend, make([]float32, count*dim))
				defer output.Close()

				backend.TimestepEmbedding(
					config,
					timesteps.DispatchPointer(),
					output.DispatchPointer(),
					count,
					dim,
					dtype.Float32,
				)
				backend.SyncDevice()

				got := downloadFloat32MetalTensor(testingObject, output)
				assertTimestepEmbeddingMetalParity(testingObject, got, want)
			})
		}
	})
}

func BenchmarkTimestepEmbeddingMetal(benchmark *testing.B) {
	backend := newMetalBenchmarkBackend(benchmark)
	defer backend.Close()

	config := device.TimestepEmbeddingConfig{
		MaxPeriod:          10000,
		DownscaleFreqShift: 0,
		TimestepDivisor:    1000,
		FlipSinToCos:       true,
	}
	timestepValues := diffusionTimestepValues(8192)

	timesteps := uploadRoPETensor(benchmark, backend, timestepValues)
	defer timesteps.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, 8192*256))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.TimestepEmbedding(
			config,
			timesteps.DispatchPointer(),
			output.DispatchPointer(),
			8192,
			256,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
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

func assertTimestepEmbeddingMetalParity(
	testingObject *testing.T,
	got []float32,
	want []float32,
) {
	testingObject.Helper()

	if len(got) != len(want) {
		testingObject.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if timestepEmbeddingMetalLaneMatches(got[index], want[index]) {
			continue
		}

		testingObject.Fatalf(
			"lane %d got=%g want=%g ulp=%d max=%d",
			index,
			got[index],
			want[index],
			cpuparity.Float32ULPDistance(got[index], want[index]),
			timestepEmbeddingMetalMaxULP,
		)
	}
}

func timestepEmbeddingMetalLaneMatches(got float32, want float32) bool {
	return cpuparity.Float32ULPDistance(got, want) <= timestepEmbeddingMetalMaxULP
}

func diffusionTimestepValues(count int) []float32 {
	timesteps := make([]float32, count)

	if count == 1 {
		timesteps[0] = 1000
		return timesteps
	}

	for index := range timesteps {
		ratio := float32(index) / float32(count-1)
		timesteps[index] = 1000 * (1 - ratio)
	}

	return timesteps
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
