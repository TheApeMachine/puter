package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/internal/scalar"
)

/*
Sum writes the elementwise sum of `values` into `*dst`.

Zero-host-sync (ARCHITECTURE.md §2.2): the scalar result is written to
the caller's destination slot rather than returned.
*/
func (reduction Reduction) Sum(dst, values unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchSum(values, count, format), format)
}

func (reduction Reduction) Prod(dst, values unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchProd(values, count, format), format)
}

func (reduction Reduction) ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchReduceMin(values, count, format), format)
}

func (reduction Reduction) ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchReduceMax(values, count, format), format)
}

func (reduction Reduction) L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchL1Norm(values, count, format), format)
}
