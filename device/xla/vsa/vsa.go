package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
VSA implements device.VSA for the XLA backend.
*/
type VSA struct {
	host Host
}

/*
Host is the XLA dispatch surface vsa operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchBind(left, right, output unsafe.Pointer, count int, format dtype.DType)
	DispatchBundle(left, right, output unsafe.Pointer, count int, format dtype.DType)
	DispatchSimilarity(left, right unsafe.Pointer, count int, format dtype.DType) float32
	DispatchPermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	DispatchInversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	NotImplemented(string)
}

/*
New wires a VSA receiver to its XLA dispatch host.
*/
func New(host Host) VSA {
	return VSA{host: host}
}

func (receiver *VSA) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *VSA) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
