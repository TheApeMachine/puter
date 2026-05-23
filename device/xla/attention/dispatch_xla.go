//go:build xla

package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (attention *Attention) ScaledDotProductAttention( config FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType, ) {
	attention.unimplemented("ScaledDotProductAttention")
}

func (attention *Attention) FlashAttention( config FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType, ) {
	attention.unimplemented("FlashAttention")
}

func (attention *Attention) MultiHeadAttention( config MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType, ) {
	attention.unimplemented("MultiHeadAttention")
}

