package rope

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestRunRoPELlama3HalfModeParity(t *testing.T) {
	convey.Convey("Given Llama 3 scaled RoPE in half mode", t, func() {
		config := llama3HalfModeRoPEConfig()
		numHeads := 2
		headDim := 8

		for _, seqLen := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar reference for N=%d", seqLen), func() {
				input := randomRoPEInput(seqLen*numHeads*headDim, 0x7300+int64(seqLen))
				want := make([]float32, len(input))
				got := make([]float32, len(input))

				referenceRoPE(want, input, config, seqLen, numHeads, headDim)
				Default.RoPE(
					config,
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&got[0]),
					seqLen,
					numHeads,
					headDim,
					dtype.Float32,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func BenchmarkRunRoPELlama3HalfMode(b *testing.B) {
	config := llama3HalfModeRoPEConfig()
	seqLen := 1024
	numHeads := 16
	headDim := 64
	input := randomRoPEInput(seqLen*numHeads*headDim, 0x7400)
	output := make([]float32, len(input))

	b.ReportAllocs()

	for b.Loop() {
		Default.RoPE(
			config,
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&output[0]),
			seqLen,
			numHeads,
			headDim,
			dtype.Float32,
		)
	}
}

func llama3HalfModeRoPEConfig() RoPEConfig {
	return RoPEConfig{
		BaseFreq:        500000.0,
		StartPosition:   3,
		Mode:            device.RoPEModeHalf,
		Scaling:         device.RoPEScalingLlama3,
		ScalingFactor:   32.0,
		LowFreqFactor:   1.0,
		HighFreqFactor:  4.0,
		OriginalContext: 8192,
	}
}

func randomRoPEInput(count int, seed int64) []float32 {
	random := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = float32((random.Float64() - 0.5) * 4)
	}

	return values
}

func referenceRoPE(
	output, input []float32,
	config RoPEConfig,
	seqLen, numHeads, headDim int,
) {
	halfDim := headDim / 2

	for seqIndex := 0; seqIndex < seqLen; seqIndex++ {
		position := float64(seqIndex + config.StartPosition)

		for headIndex := 0; headIndex < numHeads; headIndex++ {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim

			for pairIndex := 0; pairIndex < halfDim; pairIndex++ {
				cosTheta, sinTheta := referenceRoPEAngle(config, position, pairIndex, headDim)
				evenIndex, oddIndex := referenceRoPEPairIndices(config, rowOffset, halfDim, pairIndex)
				even := input[evenIndex]
				odd := input[oddIndex]
				output[evenIndex] = even*cosTheta - odd*sinTheta
				output[oddIndex] = even*sinTheta + odd*cosTheta
			}
		}
	}
}

func referenceRoPEAngle(
	config RoPEConfig,
	position float64,
	pairIndex, headDim int,
) (float32, float32) {
	exponent := -float64(2*pairIndex) / float64(headDim)
	inverseFrequency := math.Pow(config.BaseFreq, exponent)

	if config.Scaling == device.RoPEScalingLlama3 {
		inverseFrequency = referenceLlama3ScaledInverseFrequency(config, inverseFrequency)
	}

	theta := position * inverseFrequency

	return float32(math.Cos(theta)), float32(math.Sin(theta))
}

func referenceLlama3ScaledInverseFrequency(
	config RoPEConfig,
	inverseFrequency float64,
) float64 {
	wavelength := (2.0 * math.Pi) / inverseFrequency
	lowFrequencyWavelength := float64(config.OriginalContext) / config.LowFreqFactor
	highFrequencyWavelength := float64(config.OriginalContext) / config.HighFreqFactor

	if wavelength > lowFrequencyWavelength {
		return inverseFrequency / config.ScalingFactor
	}

	if wavelength < highFrequencyWavelength {
		return inverseFrequency
	}

	smooth := (float64(config.OriginalContext)/wavelength - config.LowFreqFactor) /
		(config.HighFreqFactor - config.LowFreqFactor)

	return (1.0-smooth)*(inverseFrequency/config.ScalingFactor) + smooth*inverseFrequency
}

func referenceRoPEPairIndices(
	config RoPEConfig,
	rowOffset, halfDim, pairIndex int,
) (int, int) {
	if config.Mode == device.RoPEModeHalf {
		return rowOffset + pairIndex, rowOffset + halfDim + pairIndex
	}

	evenIndex := rowOffset + 2*pairIndex

	return evenIndex, evenIndex + 1
}
