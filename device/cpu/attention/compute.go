package attention

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchScaledDotProductAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		scaledDotProductAttention(
			config, query, key, value, output,
			seqQ, seqK, depth, valueDim, format,
		)
	default:
		panic("attention: ScaledDotProductAttention unsupported dtype")
	}
}

func dispatchMultiHeadAttention(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		multiHeadAttentionSlices(
			config, query, key, value, output,
			seqQ, seqK, kvHeads, format,
		)
	default:
		panic("attention: MultiHeadAttention unsupported dtype")
	}
}

func scaledDotProductAttention(
	config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		runFlashAttentionRowTyped(
			query, key, value, output,
			rowIndex, seqQ, seqK, depth, valueDim, scale, config.Causal, format,
		)
	}
}

func multiHeadAttentionSlices(
	config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, kvHeads int,
	format dtype.DType,
) {
	scale := float32(1.0 / math.Sqrt(float64(config.HeadDim)))
	headsPerKVHead := config.NumHeads / kvHeads

	for headIndex := 0; headIndex < config.NumHeads; headIndex++ {
		kvHeadIndex := headIndex / headsPerKVHead

		runSingleHeadTyped(
			query, key, value, output,
			seqQ, seqK,
			config.HeadDim, config.NumHeads, kvHeads,
			headIndex, kvHeadIndex,
			scale, config, format,
		)
	}
}
