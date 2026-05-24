//go:build darwin && cgo

package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Dot writes the inner product of `left` and `right` into `*dst`
(ARCHITECTURE.md §2.2). The Metal host implementation currently computes
on device and reads back internally; once the static memory planner
lands (GAPS.md P1) the host signature will take the workspace-resolved
MetalBufferRef as the output, eliminating the device→host round-trip.
*/
func (product *Product) Dot(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	*(*float32)(dst) = product.host.DotProduct(left, right, count, format)
}
