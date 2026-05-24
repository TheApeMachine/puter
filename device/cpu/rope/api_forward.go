package rope

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultRotaryEmbedding = New()

func RoPE(config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType) {
	defaultRotaryEmbedding.RoPE(config, input, output, seqLen, numHeads, headDim, format)
}

func RoPEPairs(output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType) {
	defaultRotaryEmbedding.RoPEPairs(output, input, cosBuffer, sinBuffer, halfDim, format)
}
