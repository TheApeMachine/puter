//go:build cuda

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa *VSA) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return vsa.host.DispatchSimilarity(left, right, count, format)
}
