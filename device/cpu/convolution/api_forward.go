package convolution

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultConvolution = New()

func Conv1D(config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType) {
	defaultConvolution.Conv1D(config, input, weight, bias, output, batch, inChannels, inLength, outChannels, kernelLength, outLength, format)
}

func Conv2D(config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType) {
	defaultConvolution.Conv2D(config, input, weight, bias, output, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth, format)
}

func Conv3D(config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType) {
	defaultConvolution.Conv3D(config, input, weight, bias, output, batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW, format)
}

func ConvTranspose2D(config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType) {
	defaultConvolution.ConvTranspose2D(config, input, weight, bias, output, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth, format)
}
