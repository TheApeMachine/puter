//go:build cuda

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Similarity writes the dot-product similarity into `*dst`
(ARCHITECTURE.md §2.2). The CUDA host currently computes on device and
reads back internally; once the static memory planner lands
(GAPS.md P1) the host signature will take a device pointer directly.
*/
func (vsa *VSA) Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	*(*float32)(dst) = vsa.host.DispatchSimilarity(left, right, count, format)
}
