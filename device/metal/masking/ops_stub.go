//go:build !darwin || !cgo

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
	masking.host.NeedsPlatform()
}

func (masking *Masking) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.NeedsPlatform()
}

func (masking *Masking) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	masking.host.NeedsPlatform()
}
