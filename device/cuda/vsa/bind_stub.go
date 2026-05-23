//go:build !cuda

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa *VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}
