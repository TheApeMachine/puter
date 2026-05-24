//go:build xla

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Public Reduction methods write their scalar result into `*dst` rather
than returning it (ARCHITECTURE.md §2.2). The XLA host implementation
currently materializes the result through PJRT to a host scalar and
stores it at dst; once the planner / executable cache work lands
(GAPS.md §2.5–2.6) the result will be written directly into the
caller's PjRtBuffer slot, eliminating the host transfer.
*/
func (reduction *Reduction) Sum(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelSum)
}

func (reduction *Reduction) Prod(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelProd)
}

func (reduction *Reduction) ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelMin)
}

func (reduction *Reduction) ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelMax)
}

func (reduction *Reduction) L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelL1Norm)
}
