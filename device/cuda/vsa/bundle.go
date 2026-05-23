//go:build cuda

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa *VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchBundle(left, right, output, count, format)
}
