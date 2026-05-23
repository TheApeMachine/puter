//go:build darwin && cgo

package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (attention *Attention) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	attention.host.DispatchApplyMask(input, mask, output, count, format)
}

func (attention *Attention) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	attention.host.DispatchCausalMask(output, seqQ, seqK, format)
}

func (attention *Attention) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	attention.host.DispatchALiBiBias(scores, slope, output, seqQ, seqK, format)
}
