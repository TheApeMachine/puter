package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func (backend *Backend) Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.Sum(values, count, format)
}

func (backend *Backend) Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.Prod(values, count, format)
}

func (backend *Backend) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.ReduceMin(values, count, format)
}

func (backend *Backend) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.ReduceMax(values, count, format)
}

func (backend *Backend) L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.L1Norm(values, count, format)
}

func (backend *Backend) Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return dot.Dot(left, right, count, format)
}

func (backend *Backend) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	matmul.Matmul(out, left, right, rows, inner, cols, format)
}
