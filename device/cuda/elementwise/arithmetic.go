//go:build cuda

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (elementwise *Elementwise) Add(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinaryAdd)
}

func (elementwise *Elementwise) Sub(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinarySub)
}

func (elementwise *Elementwise) Mul(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinaryMul)
}

func (elementwise *Elementwise) Div(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinaryDiv)
}

func (elementwise *Elementwise) Max(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinaryMax)
}

func (elementwise *Elementwise) Min(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	elementwise.host.BinaryElementwise(dst, left, right, format, BinaryMin)
}
