//go:build xla

package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Dot writes the inner product of `left` and `right` into `*dst`
(ARCHITECTURE.md §2.2). The XLA host currently materializes the result
through PJRT to a host scalar and stores it at dst; once the planner /
executable cache work lands (GAPS.md §2.5–2.6) the result will be
written directly into the caller's PjRtBuffer slot.
*/
func (product *Product) Dot(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	product.host.DotProduct(dst, left, right, count, format)
}
