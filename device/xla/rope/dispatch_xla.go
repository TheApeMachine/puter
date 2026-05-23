//go:build xla

package rope

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (rotaryEmbedding *RotaryEmbedding) RoPE( config RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType, ) {
	rotaryEmbedding.unimplemented("RoPE")
}

func (rotaryEmbedding *RotaryEmbedding) RoPEPairs( output, input, cosBuffer, sinBuffer unsafe.Pointer, halfDim int, format dtype.DType, ) {
	rotaryEmbedding.unimplemented("RoPEPairs")
}

