//go:build !xla

package vsa

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (vsa *VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}

func (vsa *VSA) Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
	*(*float32)(dst) = 0
}

func (vsa *VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}

func (vsa *VSA) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}

func (vsa *VSA) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}

