//go:build !darwin || !cgo

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa *VSA) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	vsa.stubHost()
	return 0
}
