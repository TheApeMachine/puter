//go:build !darwin || !cgo

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa *VSA) Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}
