package layernorm

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultNorm = New()

func LayerNorm(input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType) {
	defaultNorm.LayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func RMSNorm(input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType) {
	defaultNorm.RMSNorm(input, scale, output, rows, lastDim, format)
}
