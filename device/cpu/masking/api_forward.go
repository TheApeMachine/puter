package masking

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultMasking = New()

func ALiBiBias(scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType) {
	defaultMasking.ALiBiBias(scores, slope, output, seqQ, seqK, format)
}

func ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	defaultMasking.ApplyMask(input, mask, output, count, format)
}

func CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	defaultMasking.CausalMask(output, seqQ, seqK, format)
}
