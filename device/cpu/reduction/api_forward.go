package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultReduction = New()

func Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultReduction.Sum(values, count, format)
}

func Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultReduction.Prod(values, count, format)
}

func ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultReduction.ReduceMin(values, count, format)
}

func ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultReduction.ReduceMax(values, count, format)
}

func L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultReduction.L1Norm(values, count, format)
}
