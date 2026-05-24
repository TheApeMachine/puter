package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (reduction Reduction) Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchSum(values, count, format)
}

func (reduction Reduction) Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchProd(values, count, format)
}

func (reduction Reduction) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchReduceMin(values, count, format)
}

func (reduction Reduction) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchReduceMax(values, count, format)
}

func (reduction Reduction) L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchL1Norm(values, count, format)
}
