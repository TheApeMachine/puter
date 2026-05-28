//go:build xla

package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (vsa *VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchBundle(left, right, output, count, format)
}

func (vsa *VSA) Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchSimilarity(dst, left, right, count, format)
}

func (vsa *VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchBind(left, right, output, count, format)
}

func (vsa *VSA) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchPermute(config, input, output, count, format)
}

func (vsa *VSA) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.host.DispatchInversePermute(config, input, output, count, format)
}
