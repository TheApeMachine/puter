package rope

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
RotaryEmbedding implements device.RotaryEmbedding for the XLA backend.
*/
type RotaryEmbedding struct {
	host Host
}

/*
Host is the XLA dispatch surface rope operations call into.
*/
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

/*
New wires a RotaryEmbedding receiver to its XLA dispatch host.
*/
func New(host Host) RotaryEmbedding {
	return RotaryEmbedding{host: host}
}

func (receiver *RotaryEmbedding) stubHost() {
	receiver.host.NeedsPlatform()
}
