//go:build xla

package rope

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (rotaryEmbedding *RotaryEmbedding) RoPE(
	config device.RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	rotaryEmbedding.host.DispatchRoPE(config, input, output, seqLen, numHeads, headDim, format)
}

func (rotaryEmbedding *RotaryEmbedding) RoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType,
) {
	rotaryEmbedding.host.DispatchRoPEPairs(output, input, cosBuffer, sinBuffer, halfDim, format)
}

func (rotaryEmbedding *RotaryEmbedding) MultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	rotaryEmbedding.host.DispatchMultiAxisRoPE(
		config,
		input, output,
		batch, seqLen, numHeads, headDim,
		format,
	)
}
