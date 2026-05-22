//go:build amd64

package attention

import (
	"math"

	"golang.org/x/sys/cpu"
)

func flashAttentionOnlineUpdateGeneric(
	acc, valueRow []float32,
	alpha, shifted float32,
) {
	for dimIndex := range acc {
		acc[dimIndex] = acc[dimIndex]*alpha + valueRow[dimIndex]*shifted
	}
}

func flashAttentionScaleGeneric(
	out, acc []float32,
	invNormalizer float32,
) {
	for dimIndex := range out {
		out[dimIndex] = acc[dimIndex] * invNormalizer
	}
}

func flashAttentionOnlineUpdateNative(
	acc, valueRow []float32,
	alpha, shifted float32,
) {
	length := len(acc)

	if length == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		blockCount := length &^ 15

		if blockCount > 0 {
			flashAttentionOnlineUpdateAVX512(
				&acc[0], &valueRow[0],
				alpha, shifted, blockCount,
			)
		}

		if blockCount < length {
			flashAttentionOnlineUpdateGeneric(
				acc[blockCount:],
				valueRow[blockCount:],
				alpha, shifted,
			)
		}

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		flashAttentionOnlineUpdateAVX2(
			&acc[0], &valueRow[0],
			alpha, shifted, length,
		)

		return
	}

	if cpu.X86.HasSSE2 {
		flashAttentionOnlineUpdateSSE2(
			&acc[0], &valueRow[0],
			alpha, shifted, length,
		)

		return
	}

	flashAttentionOnlineUpdateGeneric(acc, valueRow, alpha, shifted)
}

func flashAttentionScaleNative(
	out, acc []float32,
	invNormalizer float32,
) {
	length := len(out)

	if length == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		blockCount := length &^ 15

		if blockCount > 0 {
			flashAttentionScaleAVX512(
				&out[0], &acc[0],
				invNormalizer, blockCount,
			)
		}

		if blockCount < length {
			flashAttentionScaleGeneric(
				out[blockCount:],
				acc[blockCount:],
				invNormalizer,
			)
		}

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		flashAttentionScaleAVX2(
			&out[0], &acc[0],
			invNormalizer, length,
		)

		return
	}

	if cpu.X86.HasSSE2 {
		flashAttentionScaleSSE2(
			&out[0], &acc[0],
			invNormalizer, length,
		)

		return
	}

	flashAttentionScaleGeneric(out, acc, invNormalizer)
}

func RunFlashAttentionRowNative(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causalMask bool,
) {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causalMask && keyIndex > rowIndex {
			continue
		}

		queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]
		keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]
		score := DotFloat32Native(queryRow, keyRow) * scale
		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := flashExpFloat32(oldMax - maxScore)
		shifted := flashExpFloat32(score - maxScore)
		normalizer = normalizer*alpha + shifted

		valueRow := valueView[keyIndex*valueDim : (keyIndex+1)*valueDim]
		flashAttentionOnlineUpdateNative(accumulator, valueRow, alpha, shifted)
	}

	outputRow := outView[rowIndex*valueDim : (rowIndex+1)*valueDim]

	if normalizer == 0 {
		for dimIndex := range outputRow {
			outputRow[dimIndex] = 0
		}

		return
	}

	invNormalizer := float32(1) / normalizer
	flashAttentionScaleNative(outputRow, accumulator, invNormalizer)
}
