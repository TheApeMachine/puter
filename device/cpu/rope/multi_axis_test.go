package rope

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

func TestRunMultiAxisRoPEFloat32Parity(t *testing.T) {
	convey.Convey("Given multi-axis RoPE float32 inputs", t, func() {
		batch := 1
		numHeads := 2
		headDim := 16

		for _, seqLen := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the scalar reference for N=%d", seqLen), func() {
				config := multiAxisRoPEConfig(seqLen)
				input := randomRoPEInput(batch*seqLen*numHeads*headDim, 0x9500+int64(seqLen))
				want := make([]float32, len(input))
				got := make([]float32, len(input))

				referenceMultiAxisRoPE(want, input, config, batch, seqLen, numHeads, headDim)
				Default.MultiAxisRoPE(
					config,
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&got[0]),
					batch,
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

func BenchmarkRunMultiAxisRoPEFloat32(benchmark *testing.B) {
	batch := 1
	seqLen := 5120
	numHeads := 24
	headDim := 128
	config := device.MultiAxisRoPEConfig{
		BaseFreq:     10000,
		LatentSeqLen: 4096,
		LatentSide:   64,
		AxisCount:    4,
		AxisDim0:     32,
		AxisDim1:     32,
		AxisDim2:     32,
		AxisDim3:     32,
	}
	input := randomRoPEInput(batch*seqLen*numHeads*headDim, 0x9501)
	output := make([]float32, len(input))

	benchmark.SetBytes(int64(len(input) * 4 * 2))
	benchmark.ReportAllocs()

	for benchmark.Loop() {
		Default.MultiAxisRoPE(
			config,
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&output[0]),
			batch,
			seqLen,
			numHeads,
			headDim,
			dtype.Float32,
		)
	}
}

func multiAxisRoPEConfig(seqLen int) device.MultiAxisRoPEConfig {
	latentSeqLen := max(seqLen/2, 1)
	latentSide := int(math.Ceil(math.Sqrt(float64(latentSeqLen))))

	return device.MultiAxisRoPEConfig{
		BaseFreq:     2000,
		LatentSeqLen: latentSeqLen,
		LatentSide:   latentSide,
		AxisCount:    4,
		AxisDim0:     2,
		AxisDim1:     6,
		AxisDim2:     4,
		AxisDim3:     4,
	}
}

func referenceMultiAxisRoPE(
	output, input []float32,
	config device.MultiAxisRoPEConfig,
	batch, seqLen, numHeads, headDim int,
) {
	halfDim := headDim / 2

	for batchIndex := range batch {
		for seqIndex := range seqLen {
			for headIndex := range numHeads {
				rowOffset := ((batchIndex*seqLen+seqIndex)*numHeads + headIndex) * headDim

				for pairIndex := range halfDim {
					cosTheta, sinTheta := referenceMultiAxisRoPEAngle(config, seqIndex, pairIndex, seqLen, headDim)
					evenIndex := rowOffset + pairIndex*2
					oddIndex := evenIndex + 1
					even := input[evenIndex]
					odd := input[oddIndex]
					output[evenIndex] = even*cosTheta - odd*sinTheta
					output[oddIndex] = even*sinTheta + odd*cosTheta
				}
			}
		}
	}
}

func referenceMultiAxisRoPEAngle(
	config device.MultiAxisRoPEConfig,
	seqIndex, pairIndex, seqLen, headDim int,
) (float32, float32) {
	textLen := max(seqLen-config.LatentSeqLen, 0)
	axisIndex, localPair, axisPairCount := referenceMultiAxisRoPEAxis(config, pairIndex)

	position := referenceMultiAxisRoPEPosition(config, seqIndex, textLen, axisIndex)
	axisDim := float64(axisPairCount * 2)

	if axisDim == 0 {
		return 1, 0
	}

	exponent := -2.0 * float64(localPair) / axisDim
	angle := float64(position) * math.Pow(config.BaseFreq, exponent)

	return float32(math.Cos(angle)), float32(math.Sin(angle))
}

func referenceMultiAxisRoPEAxis(
	config device.MultiAxisRoPEConfig,
	pairIndex int,
) (int, int, int) {
	axisDims := []int{config.AxisDim0, config.AxisDim1, config.AxisDim2, config.AxisDim3}
	pairStart := 0

	for axisIndex := range config.AxisCount {
		axisPairCount := axisDims[axisIndex] / 2
		pairEnd := pairStart + axisPairCount

		if pairIndex < pairEnd {
			return axisIndex, pairIndex - pairStart, axisPairCount
		}

		pairStart = pairEnd
	}

	return 0, pairIndex, 0
}

func referenceMultiAxisRoPEPosition(
	config device.MultiAxisRoPEConfig,
	seqIndex, textLen, axisIndex int,
) int {
	if seqIndex < textLen {
		if config.AxisCount == 4 && axisIndex == 3 {
			return seqIndex
		}

		if config.AxisCount < 4 && axisIndex == 0 {
			return seqIndex
		}

		return 0
	}

	imageIndex := seqIndex - textLen

	switch axisIndex {
	case 1:
		return imageIndex / config.LatentSide
	case 2:
		return imageIndex % config.LatentSide
	default:
		return 0
	}
}
