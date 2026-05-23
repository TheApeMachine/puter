package cuda

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (backend *Backend) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	backend.Attention.ApplyMask(input, mask, output, count, format)
}

func (backend *Backend) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	backend.Attention.CausalMask(output, seqQ, seqK, format)
}

func (backend *Backend) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	backend.Attention.ALiBiBias(scores, slope, output, seqQ, seqK, format)
}
