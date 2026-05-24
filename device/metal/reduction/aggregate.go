//go:build darwin && cgo

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Public Reduction methods write their scalar result into `*dst` rather
than returning it (ARCHITECTURE.md §2.2). The Metal host implementation
still computes the value on device and reads it back internally; once
the static memory planner lands (GAPS.md P1) the host signature will
be migrated to take a workspace-resolved MetalBufferRef as the output,
eliminating the device→host round-trip entirely.
*/
func (reduction *Reduction) Sum(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelSum)
}

func (reduction *Reduction) Prod(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelProd)
}

func (reduction *Reduction) ReduceMin(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelMin)
}

func (reduction *Reduction) ReduceMax(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelMax)
}

func (reduction *Reduction) L1Norm(
	dst, values unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = reduction.host.ReductionScalar(values, count, format, KernelL1Norm)
}
