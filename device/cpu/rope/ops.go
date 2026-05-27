package rope

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func requireRoPEFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("rope: RoPEPairs unsupported dtype")
	}
}

func (rotaryEmbedding RotaryEmbedding) RoPE(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	dispatchRoPE(config, input, output, seqLen, numHeads, headDim, format)
}

func (rotaryEmbedding RotaryEmbedding) RoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType,
) {
	requireRoPEFloat32(format)

	RopePairsNative(
		unsafe.Slice((*float32)(output), halfDim*2),
		unsafe.Slice((*float32)(input), halfDim*2),
		unsafe.Slice((*float32)(cosBuffer), halfDim),
		unsafe.Slice((*float32)(sinBuffer), halfDim),
	)
}

func (rotaryEmbedding RotaryEmbedding) MultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	dispatchMultiAxisRoPE(config, input, output, batch, seqLen, numHeads, headDim, format)
}
