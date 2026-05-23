//go:build xla

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (reduction *Reduction) Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.host.ReductionScalar(values, count, format, KernelSum)
}

func (reduction *Reduction) Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.host.ReductionScalar(values, count, format, KernelProd)
}

func (reduction *Reduction) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.host.ReductionScalar(values, count, format, KernelMin)
}

func (reduction *Reduction) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.host.ReductionScalar(values, count, format, KernelMax)
}

func (reduction *Reduction) L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	return reduction.host.ReductionScalar(values, count, format, KernelL1Norm)
}
