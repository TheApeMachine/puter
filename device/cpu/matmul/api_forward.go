package matmul

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultGemm = New()

func Matmul(out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType) {
	defaultGemm.Matmul(out, left, right, rows, inner, cols, format)
}
