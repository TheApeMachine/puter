package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Dot writes the inner product of `left` and `right` into `*dst`. Zero-host-sync
per ARCHITECTURE.md §2.2.
*/
func (product Product) Dot(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = dispatchDot(left, right, count, format)
}
