package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/masking"
)

func (backend *Backend) ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	masking.ALiBiBias(scores, slope, output, seqQ, seqK, format)
}

func (backend *Backend) CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	masking.CausalMask(output, seqQ, seqK, format)
}

func (backend *Backend) ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	masking.ApplyMask(input, mask, output, count, format)
}
