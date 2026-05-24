package attention

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultAttention = New()

func FlashAttention(config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType) {
	defaultAttention.FlashAttention(config, query, key, value, output, seqQ, seqK, depth, valueDim, format)
}

func MultiHeadAttention(config MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType) {
	defaultAttention.MultiHeadAttention(config, query, key, value, output, seqQ, seqK, format)
}

func ScaledDotProductAttention(config FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType) {
	defaultAttention.ScaledDotProductAttention(config, query, key, value, output, seqQ, seqK, depth, valueDim, format)
}
