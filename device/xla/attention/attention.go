package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Attention implements device.Attention for the XLA backend.
*/
type Attention struct {
	host Host
}

/*
Host is the XLA dispatch surface attention operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchScaledDotProductAttention(
		config device.FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
	DispatchMultiHeadAttention(
		config device.MultiHeadAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK int,
		format dtype.DType,
	)
}

/*
New wires a Attention receiver to its XLA dispatch host.
*/
func New(host Host) Attention {
	return Attention{host: host}
}

func (receiver *Attention) stubHost() {
	receiver.host.NeedsPlatform()
}
