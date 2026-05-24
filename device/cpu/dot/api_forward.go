package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultProduct = New()

func Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultProduct.Dot(left, right, count, format)
}
