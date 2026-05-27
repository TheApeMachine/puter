package rope

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func dispatchRoPE(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	if seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	if err := config.Validate(); err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		runRoPE(config, input, output, seqLen, numHeads, headDim, format)
	default:
		panic("rope: unsupported dtype")
	}
}

func runRoPE(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	halfDim := headDim / 2

	cosBuffer := BorrowFloat32Buffer(halfDim)
	sinBuffer := BorrowFloat32Buffer(halfDim)

	defer ReleaseFloat32Buffer(cosBuffer)
	defer ReleaseFloat32Buffer(sinBuffer)

	for seqIndex := 0; seqIndex < seqLen; seqIndex++ {
		position := float64(seqIndex + config.StartPosition)

		for pairIndex := 0; pairIndex < halfDim; pairIndex++ {
			theta := position * ropeInverseFrequency(config, pairIndex, headDim)
			cosBuffer[pairIndex] = float32(math.Cos(theta))
			sinBuffer[pairIndex] = float32(math.Sin(theta))
		}

		for headIndex := 0; headIndex < numHeads; headIndex++ {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim

			if format == dtype.Float32 {
				outputView := unsafe.Slice((*float32)(output), seqLen*numHeads*headDim)
				inputView := unsafe.Slice((*float32)(input), seqLen*numHeads*headDim)
				ropePairsFloat32(config, outputView, inputView, rowOffset, cosBuffer, sinBuffer)
				continue
			}

			load, store := ropeLoadStore(format)
			ropePairsTyped(
				config, output, input, rowOffset,
				cosBuffer, sinBuffer, halfDim,
				load, store,
			)
		}
	}
}

func ropeInverseFrequency(config RoPEConfig, pairIndex, headDim int) float64 {
	exponent := -float64(2*pairIndex) / float64(headDim)
	inverseFrequency := math.Pow(config.BaseFreq, exponent)

	if config.Scaling != device.RoPEScalingLlama3 {
		return inverseFrequency
	}

	return llama3ScaledInverseFrequency(config, inverseFrequency)
}

func llama3ScaledInverseFrequency(config RoPEConfig, inverseFrequency float64) float64 {
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

func ropePairsFloat32(
	config RoPEConfig,
	outputView, inputView []float32,
	rowOffset int,
	cosBuffer, sinBuffer []float32,
) {
	switch config.Mode {
	case device.RoPEModeInterleaved:
		RopePairsNative(
			outputView[rowOffset:rowOffset+len(cosBuffer)*2],
			inputView[rowOffset:rowOffset+len(cosBuffer)*2],
			cosBuffer,
			sinBuffer,
		)
	case device.RoPEModeHalf:
		ropePairsHalfFloat32(outputView, inputView, rowOffset, cosBuffer, sinBuffer)
	}
}

func ropePairsHalfFloat32(
	outputView, inputView []float32,
	rowOffset int,
	cosBuffer, sinBuffer []float32,
) {
	halfDim := len(cosBuffer)

	for pairIndex, cos := range cosBuffer {
		sin := sinBuffer[pairIndex]
		evenIndex := rowOffset + pairIndex
		oddIndex := rowOffset + halfDim + pairIndex
		even := inputView[evenIndex]
		odd := inputView[oddIndex]
		outputView[evenIndex] = even*cos - odd*sin
		outputView[oddIndex] = even*sin + odd*cos
	}
}

func ropePairsTyped(
	config RoPEConfig,
	output, input unsafe.Pointer,
	rowOffset int,
	cosBuffer, sinBuffer []float32,
	halfDim int,
	load ropeLoadFunc,
	store ropeStoreFunc,
) {
	for pairIndex := 0; pairIndex < halfDim; pairIndex++ {
		cos := cosBuffer[pairIndex]
		sin := sinBuffer[pairIndex]
		evenIndex, oddIndex := ropePairIndices(config, rowOffset, halfDim, pairIndex)
		even := load(input, evenIndex)
		odd := load(input, oddIndex)
		store(output, evenIndex, even*cos-odd*sin)
		store(output, oddIndex, even*sin+odd*cos)
	}
}

func ropePairIndices(config RoPEConfig, rowOffset, halfDim, pairIndex int) (int, int) {
	if config.Mode == device.RoPEModeHalf {
		return rowOffset + pairIndex, rowOffset + halfDim + pairIndex
	}

	evenIndex := rowOffset + 2*pairIndex

	return evenIndex, evenIndex + 1
}
