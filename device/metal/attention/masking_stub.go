//go:build !darwin || !cgo

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
	attention.stubHost()
}

func (attention *Attention) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	attention.stubHost()
}

func (attention *Attention) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	attention.stubHost()
}
