package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

type Attention struct {
	host Host
}

func New(host Host) Attention {
	return Attention{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchFlashAttention(
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
	DispatchScaledDotProductAttention(
		config device.FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
}
