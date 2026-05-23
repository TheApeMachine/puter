package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Convolution implements device.Convolution for the Metal backend.
*/
type Convolution struct {
	host Host
}

func New(host Host) Convolution {
	return Convolution{host: host}
}

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
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW int,
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
}
