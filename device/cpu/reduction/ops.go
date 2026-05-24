package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Sum writes the elementwise sum of `values` into `*dst`.

Zero-host-sync (ARCHITECTURE.md §2.2): the scalar result is written to
the caller's destination slot rather than returned. For the CPU backend
the workspace lives in host RAM, so the store is a direct float32 write
at `dst`; for Metal/CUDA/XLA the equivalent method writes through the
device-resident workspace pointer.
*/
func (reduction Reduction) Sum(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchSum(values, count, format)
}

func (reduction Reduction) Prod(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchProd(values, count, format)
}

func (reduction Reduction) ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchReduceMin(values, count, format)
}

func (reduction Reduction) ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchReduceMax(values, count, format)
}

func (reduction Reduction) L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchL1Norm(values, count, format)
}
