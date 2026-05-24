package dequant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Dequantization implements device.Dequantization for the XLA backend.
*/
type Dequantization struct {
	host Host
}

/*
Host is the XLA dispatch surface dequant operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchDequant(
		dst, src unsafe.Pointer,
		count int,
		config device.DequantInt8Config,
		dstFormat, srcFormat dtype.DType,
	)
	DispatchDequant4(
		dst, src unsafe.Pointer,
		elementCount int,
		config device.DequantInt4Config,
		dstFormat, srcFormat dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Dequantization receiver to its XLA dispatch host.
*/
func New(host Host) Dequantization {
	return Dequantization{host: host}
}

func (receiver *Dequantization) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Dequantization) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
