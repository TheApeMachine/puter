//go:build darwin && cgo

package masking

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (masking *Masking) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	masking.host.DispatchApplyMask(input, mask, output, count, format)
}

func (masking *Masking) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.DispatchCausalMask(output, seqQ, seqK, format)
}

func (masking *Masking) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.DispatchALiBiBias(scores, slope, output, seqQ, seqK, format)
}
