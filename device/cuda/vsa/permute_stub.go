//go:build !cuda

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (vsa *VSA) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}

func (vsa *VSA) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.stubHost()
}
