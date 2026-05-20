package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/elementwise"
)

func (backend *Backend) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Add(dst, left, right, count, format)
}

func (backend *Backend) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Sub(dst, left, right, count, format)
}

func (backend *Backend) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Mul(dst, left, right, count, format)
}

func (backend *Backend) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Div(dst, left, right, count, format)
}

func (backend *Backend) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Max(dst, left, right, count, format)
}

func (backend *Backend) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Min(dst, left, right, count, format)
}

func (backend *Backend) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Abs(dst, src, count, format)
}

func (backend *Backend) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Neg(dst, src, count, format)
}

func (backend *Backend) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.Sqrt(dst, src, count, format)
}

func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementwise.ReLU(dst, src, count, format)
}

func (backend *Backend) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	elementwise.Axpy(y, x, count, alpha, format)
}
