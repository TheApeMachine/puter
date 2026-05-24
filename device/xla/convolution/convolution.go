package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Convolution implements device.Convolution for the XLA backend.
*/
type Convolution struct {
	host Host
}

/*
Host is the XLA dispatch surface convolution operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchConv1D(
		config device.Conv1DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inLength, outChannels, kernelLength, outLength int,
		format dtype.DType,
	)
	DispatchConv2D(
		config device.Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchConv3D(
		config device.Conv3DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inDepth, inHeight, inWidth,
		outChannels, kernelDepth, kernelHeight, kernelWidth,
		outDepth, outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchConvTranspose2D(
		config device.Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Convolution receiver to its XLA dispatch host.
*/
func New(host Host) Convolution {
	return Convolution{host: host}
}

func (receiver *Convolution) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Convolution) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
