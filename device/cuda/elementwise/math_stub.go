//go:build !cuda

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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


