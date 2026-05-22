package rope

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
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
			exponent := -float64(2*pairIndex) / float64(headDim)
			theta := position * math.Pow(config.BaseFreq, exponent)
			cosBuffer[pairIndex] = float32(math.Cos(theta))
			sinBuffer[pairIndex] = float32(math.Sin(theta))
		}

		for headIndex := 0; headIndex < numHeads; headIndex++ {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim

			if format == dtype.Float32 {
				outputView := unsafe.Slice((*float32)(output), seqLen*numHeads*headDim)
				inputView := unsafe.Slice((*float32)(input), seqLen*numHeads*headDim)
				RopePairsNative(
					outputView[rowOffset:rowOffset+headDim],
					inputView[rowOffset:rowOffset+headDim],
					cosBuffer,
					sinBuffer,
				)
				continue
			}

			load, store := ropeLoadStore(format)
			ropePairsTyped(
				output, input, rowOffset,
				cosBuffer, sinBuffer, halfDim,
				load, store,
			)
		}
	}
}

func ropePairsTyped(
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
		evenIndex := rowOffset + 2*pairIndex
		oddIndex := rowOffset + 2*pairIndex + 1
		even := load(input, evenIndex)
		odd := load(input, oddIndex)
		store(output, evenIndex, even*cos-odd*sin)
		store(output, oddIndex, even*sin+odd*cos)
	}
}
