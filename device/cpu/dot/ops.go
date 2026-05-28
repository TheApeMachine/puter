package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/internal/scalar"
)

/*
Dot writes the inner product of `left` and `right` into `*dst`. Zero-host-sync
per ARCHITECTURE.md §2.2.
*/
func (product Product) Dot(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	if format == dtype.Int8 {
		scalar.StoreInt32(dst, dispatchDotInt8(left, right, count))
		return
	}

	scalar.StoreFloat32(dst, dispatchDot(left, right, count, format), format)
}
