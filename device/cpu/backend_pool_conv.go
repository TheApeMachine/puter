package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/convolution"
	"github.com/theapemachine/puter/device/cpu/pool"
)

func (backend *Backend) MaxPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.MaxPool2D(
		poolConfig(config), input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format,
	)
}

func (backend *Backend) AvgPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.AvgPool2D(
		poolConfig(config), input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format,
	)
}

func (backend *Backend) AdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.AdaptiveMaxPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (backend *Backend) AdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.AdaptiveAvgPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (backend *Backend) Conv2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	convolution.Conv2D(
		conv2DConfig(config), input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth,
		format,
	)
}

func (backend *Backend) Conv1D(
	config device.Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
) {
	convolution.Conv1D(
		conv1DConfig(config), input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
	)
}

func (backend *Backend) Conv3D(
	config device.Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
) {
	convolution.Conv3D(
		conv3DConfig(config), input, weight, bias, output,
		batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW,
		format,
	)
}

func (backend *Backend) ConvTranspose2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	convolution.ConvTranspose2D(
		conv2DConfig(config), input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth,
		format,
	)
}
