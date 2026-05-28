package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

type VSA struct {
	host Host
}

func New(host Host) VSA {
	return VSA{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchBind(left, right, output unsafe.Pointer, count int, format dtype.DType)
	DispatchBundle(left, right, output unsafe.Pointer, count int, format dtype.DType)
	DispatchInversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	DispatchPermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	DispatchSimilarity(dst, left, right unsafe.Pointer, count int, format dtype.DType)
}
