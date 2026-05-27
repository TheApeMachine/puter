package rope

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
RotaryEmbedding implements device.RoPE for the Metal backend.
*/
type RotaryEmbedding struct {
	host Host
}

func New(host Host) RotaryEmbedding {
	return RotaryEmbedding{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchRoPE(
		config device.RoPEConfig,
		input, output unsafe.Pointer,
		seqLen, numHeads, headDim int,
		format dtype.DType,
	)
	DispatchRoPEPairs(
		output, input, cosBuffer, sinBuffer unsafe.Pointer,
		halfDim int,
		format dtype.DType,
	)
	DispatchMultiAxisRoPE(
		config device.MultiAxisRoPEConfig,
		input, output unsafe.Pointer,
		batch, seqLen, numHeads, headDim int,
		format dtype.DType,
	)
}
