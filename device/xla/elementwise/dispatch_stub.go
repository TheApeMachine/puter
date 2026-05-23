//go:build !xla

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (elementwise *Elementwise) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.stubHost()
}

func (elementwise *Elementwise) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	elementwise.stubHost()
}

