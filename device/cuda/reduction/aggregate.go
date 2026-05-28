//go:build cuda

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (reduction *Reduction) Sum(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	reduction.host.ReductionScalar(dst, values, count, format, KernelSum)
}

func (reduction *Reduction) Prod(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	reduction.host.ReductionScalar(dst, values, count, format, KernelProd)
}

func (reduction *Reduction) ReduceMin(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	reduction.host.ReductionScalar(dst, values, count, format, KernelMin)
}

func (reduction *Reduction) ReduceMax(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	reduction.host.ReductionScalar(dst, values, count, format, KernelMax)
}

func (reduction *Reduction) L1Norm(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	reduction.host.ReductionScalar(dst, values, count, format, KernelL1Norm)
}
