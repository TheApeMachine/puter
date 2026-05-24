package quant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Quantization implements device.Quantization for the XLA backend.
*/
type Quantization struct {
	host Host
}

/*
Host is the XLA dispatch surface quant operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchQuant(
		dst, src unsafe.Pointer,
		count int,
		config device.DequantInt8Config,
		dstFormat, srcFormat dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Quantization receiver to its XLA dispatch host.
*/
func New(host Host) Quantization {
	return Quantization{host: host}
}

func (receiver *Quantization) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Quantization) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
