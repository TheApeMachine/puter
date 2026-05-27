package attention

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func typedRowDot(left, right unsafe.Pointer, length int, format dtype.DType) float32 {
	switch format {
	case dtype.Float32:
		leftRow := unsafe.Slice((*float32)(left), length)
		rightRow := unsafe.Slice((*float32)(right), length)
		return DotFloat32Native(leftRow, rightRow)
	case dtype.BFloat16:
		leftRow := unsafe.Slice((*dtype.BF16)(left), length)
		rightRow := unsafe.Slice((*dtype.BF16)(right), length)
		dot := DotBFloat16Native(leftRow, rightRow)
		return (&dot).Float32()
	case dtype.Float16:
		leftRow := unsafe.Slice((*dtype.F16)(left), length)
		rightRow := unsafe.Slice((*dtype.F16)(right), length)
		dot := DotFloat16Native(leftRow, rightRow)
		return dot.Float32()
	default:
		panic("attention: unsupported dtype for dot")
	}
}

func runFlashAttentionRowTyped(
	query, key, value, output unsafe.Pointer,
	rowIndex, seqQ, seqK, depth, valueDim int,
	scale float32,
	causal bool,
	format dtype.DType,
) {
	if format == dtype.Float32 {
		queryView := unsafe.Slice((*float32)(query), seqQ*depth)
		keyView := unsafe.Slice((*float32)(key), seqK*depth)
		valueView := unsafe.Slice((*float32)(value), seqK*valueDim)
		outputView := unsafe.Slice((*float32)(output), seqQ*valueDim)

		RunFlashAttentionRowNative(
			queryView, keyView, valueView, outputView,
			rowIndex, seqK, depth, valueDim, scale, causal,
		)

		return
	}

	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)
	queryRow := typedElementPointer(query, rowIndex*depth, format)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causal && keyIndex > rowIndex {
			continue
		}

		keyRow := typedElementPointer(key, keyIndex*depth, format)
		score := typedRowDot(queryRow, keyRow, depth, format) * scale
		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := flashExpFloat32(oldMax - maxScore)
		shifted := flashExpFloat32(score - maxScore)
		normalizer = normalizer*alpha + shifted

		valueBase := keyIndex * valueDim

		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			valueElement := loadTyped(value, valueBase+dimIndex, format)
			accumulator[dimIndex] = accumulator[dimIndex]*alpha + valueElement*shifted
		}
	}

	outBase := rowIndex * valueDim

	if normalizer == 0 {
		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			storeTyped(output, outBase+dimIndex, 0, format)
		}

		return
	}

	invNormalizer := float32(1) / normalizer

	for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
		storeTyped(output, outBase+dimIndex, accumulator[dimIndex]*invNormalizer, format)
	}
}

func computeAttentionScoresTyped(
	query, key unsafe.Pointer,
	seqQ, seqK, depth int,
	scale float32,
	format dtype.DType,
) []float32 {
	scores := make([]float32, seqQ*seqK)

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		queryRow := typedElementPointer(query, rowIndex*depth, format)

		for keyIndex := 0; keyIndex < seqK; keyIndex++ {
			keyRow := typedElementPointer(key, keyIndex*depth, format)
			scores[rowIndex*seqK+keyIndex] = typedRowDot(queryRow, keyRow, depth, format) * scale
		}
	}

	return scores
}

func computeWeightedOutputTyped(
	scores []float32,
	value, output unsafe.Pointer,
	seqQ, seqK, valueDim int,
	format dtype.DType,
) {
	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		scoresBase := rowIndex * seqK
		outBase := rowIndex * valueDim

		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			var sum float32

			for keyIndex := 0; keyIndex < seqK; keyIndex++ {
				valueElement := loadTyped(value, keyIndex*valueDim+dimIndex, format)
				sum += scores[scoresBase+keyIndex] * valueElement
			}

			storeTyped(output, outBase+dimIndex, sum, format)
		}
	}
}

func computeHeadScoresTyped(
	query, key unsafe.Pointer,
	qIndex, seqQ, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
	scale float32,
	scores []float32,
	config MultiHeadAttentionConfig,
	format dtype.DType,
) {
	queryHead := typedElementPointer(
		query,
		qIndex*queryStride+queryHeadOffset,
		format,
	)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		keyHead := typedElementPointer(
			key,
			keyIndex*kvStride+kvHeadOffset,
			format,
		)
		score := typedRowDot(queryHead, keyHead, headDim, format) * scale

		absoluteQueryIndex := qIndex + seqK - seqQ

		if config.Causal && keyIndex > absoluteQueryIndex {
			score = float32(math.Inf(-1))
		}

		if config.WindowSize > 0 && qIndex-keyIndex >= config.WindowSize {
			score = float32(math.Inf(-1))
		}

		if config.ALiBiSlope != 0 {
			score += config.ALiBiSlope * float32(keyIndex-qIndex)
		}

		scores[keyIndex] = score
	}
}

func writeHeadOutputTyped(
	scores []float32,
	value, output unsafe.Pointer,
	qIndex, seqK, headDim int,
	queryHeadOffset, kvHeadOffset int,
	queryStride, kvStride int,
	format dtype.DType,
) {
	outBase := qIndex*queryStride + queryHeadOffset

	for dimIndex := 0; dimIndex < headDim; dimIndex++ {
		var sum float32

		for keyIndex := 0; keyIndex < seqK; keyIndex++ {
			valueIndex := keyIndex*kvStride + kvHeadOffset + dimIndex
			sum += scores[keyIndex] * loadTyped(value, valueIndex, format)
		}

		storeTyped(output, outBase+dimIndex, sum, format)
	}
}

func runSingleHeadTyped(
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, headDim, numHeads, kvHeads, headIndex, kvHeadIndex int,
	scale float32,
	config MultiHeadAttentionConfig,
	format dtype.DType,
) {
	queryHeadOffset := headIndex * headDim
	kvHeadOffset := kvHeadIndex * headDim
	queryStride := numHeads * headDim
	kvStride := kvHeads * headDim
	scores := make([]float32, seqK)

	for qIndex := 0; qIndex < seqQ; qIndex++ {
		computeHeadScoresTyped(
			query, key,
			qIndex, seqQ, seqK, headDim,
			queryHeadOffset, kvHeadOffset,
			queryStride, kvStride,
			scale, scores,
			config, format,
		)
		StableSoftmaxRowNative(scores)
		writeHeadOutputTyped(
			scores, value, output,
			qIndex, seqK, headDim,
			queryHeadOffset, kvHeadOffset,
			queryStride, kvStride,
			format,
		)
	}
}
