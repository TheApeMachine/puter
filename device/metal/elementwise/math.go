//go:build darwin && cgo

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (elementwise *Elementwise) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	elementwise.host.UnaryElementwise(dst, src, format, UnaryAbs)
}

func (elementwise *Elementwise) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	elementwise.host.UnaryElementwise(dst, src, format, UnaryNeg)
}

func (elementwise *Elementwise) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	elementwise.host.UnaryElementwise(dst, src, format, UnarySqrt)
}

func (elementwise *Elementwise) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	elementwise.host.UnaryElementwise(dst, src, format, UnaryReLU)
}
