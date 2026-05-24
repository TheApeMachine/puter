package elementwise

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultElementwise = New()

func Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Abs(dst, src, count, format)
}

func Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Add(dst, left, right, count, format)
}

func Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	defaultElementwise.Axpy(y, x, count, alpha, format)
}

func Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Div(dst, left, right, count, format)
}

func Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Max(dst, left, right, count, format)
}

func Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Min(dst, left, right, count, format)
}

func Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Mul(dst, left, right, count, format)
}

func Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Neg(dst, src, count, format)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.ReLU(dst, src, count, format)
}

func Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Sqrt(dst, src, count, format)
}

func Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	defaultElementwise.Sub(dst, left, right, count, format)
}
