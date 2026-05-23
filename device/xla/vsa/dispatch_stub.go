//go:build !xla

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (vSA *VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vSA.stubHost()
}

func (vSA *VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vSA.stubHost()
}

func (vSA *VSA) Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vSA.stubHost()
}

func (vSA *VSA) InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vSA.stubHost()
}

