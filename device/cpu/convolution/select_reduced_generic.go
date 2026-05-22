//go:build !arm64 && !amd64

package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Conv2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	Conv2DTypedScalar(
		dtype.BFloat16,
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func Conv2DFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	Conv2DTypedScalar(
		dtype.Float16,
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func Conv1DBFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	Conv1DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength)
}

func Conv1DFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	Conv1DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength)
}

func Conv3DBFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	Conv3DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)
}

func Conv3DFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	Conv3DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)
}

func ConvTranspose2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	ConvTranspose2DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)
}

func ConvTranspose2DFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	ConvTranspose2DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)
}
