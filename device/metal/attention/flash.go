//go:build darwin && cgo

package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (attention *Attention) FlashAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	attention.host.DispatchFlashAttention(config, query, key, value, output, seqQ, seqK, depth, valueDim, format)
}
