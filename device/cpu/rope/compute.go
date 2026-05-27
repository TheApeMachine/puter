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

func dispatchMultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	if batch == 0 || seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	if err := config.Validate(); err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		runMultiAxisRoPE(config, input, output, batch, seqLen, numHeads, headDim, format)
	default:
		panic("multi-axis rope: unsupported dtype")
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

func runMultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	load, store := ropeLoadStore(format)
	halfDim := headDim / 2
	pairCount := batch * seqLen * numHeads * halfDim

	for pairOffset := range pairCount {
		indices := multiAxisRoPEIndices(pairOffset, seqLen, numHeads, headDim)
		cosTheta, sinTheta := multiAxisRoPEAngle(config, indices.seqIndex, indices.pairIndex, seqLen, headDim)
		even := load(input, indices.evenIndex)
		odd := load(input, indices.oddIndex)

		store(output, indices.evenIndex, even*cosTheta-odd*sinTheta)
		store(output, indices.oddIndex, even*sinTheta+odd*cosTheta)
	}
}

type multiAxisRoPEIndex struct {
	seqIndex  int
	pairIndex int
	evenIndex int
	oddIndex  int
}

func multiAxisRoPEIndices(pairOffset, seqLen, numHeads, headDim int) multiAxisRoPEIndex {
	halfDim := headDim / 2
	pairIndex := pairOffset % halfDim
	headIndex := (pairOffset / halfDim) % numHeads
	seqIndex := (pairOffset / (halfDim * numHeads)) % seqLen
	batchIndex := pairOffset / (halfDim * numHeads * seqLen)
	headOffset := ((batchIndex*seqLen+seqIndex)*numHeads + headIndex) * headDim
	evenIndex := headOffset + pairIndex*2

	return multiAxisRoPEIndex{
		seqIndex:  seqIndex,
		pairIndex: pairIndex,
		evenIndex: evenIndex,
		oddIndex:  evenIndex + 1,
	}
}

func multiAxisRoPEAngle(
	config device.MultiAxisRoPEConfig,
	seqIndex, pairIndex, seqLen, headDim int,
) (float32, float32) {
	halfDim := headDim / 2
	textLen := max(seqLen-config.LatentSeqLen, 0)
	axisPairCount := halfDim / 4
	axisIndex := 0
	localPair := pairIndex

	if axisPairCount > 0 {
		axisIndex = pairIndex / axisPairCount
		localPair = pairIndex - axisIndex*axisPairCount
	}

	position := multiAxisRoPEPosition(config, seqIndex, textLen, axisIndex)
	axisDim := float64(axisPairCount * 2)

	if axisDim == 0 {
		return 1, 0
	}

	exponent := -2.0 * float64(localPair) / axisDim
	angle := float64(position) * math.Pow(config.BaseFreq, exponent)

	return float32(math.Cos(angle)), float32(math.Sin(angle))
}

func multiAxisRoPEPosition(
	config device.MultiAxisRoPEConfig,
	seqIndex, textLen, axisIndex int,
) int {
	if seqIndex < textLen {
		if axisIndex == 3 {
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
