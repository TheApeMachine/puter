package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func runConv2DReduced(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	switch format {
	case dtype.BFloat16:
		Conv2DBFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	case dtype.Float16:
		Conv2DFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	default:
		Conv2DTypedScalar(
			format,
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	}
}

func runConv1DReduced(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
) {
	switch format {
	case dtype.BFloat16:
		Conv1DBFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)
	case dtype.Float16:
		Conv1DFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)
	default:
		Conv1DTypedScalar(
			format,
			config,
			input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)
	}
}

func runConv3DReduced(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
) {
	switch format {
	case dtype.BFloat16:
		Conv3DBFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)
	case dtype.Float16:
		Conv3DFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)
	default:
		Conv3DTypedScalar(
			format,
			config,
			input, weight, bias, output,
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)
	}
}

func runConvTranspose2DReduced(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	switch format {
	case dtype.BFloat16:
		ConvTranspose2DBFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	case dtype.Float16:
		ConvTranspose2DFloat16Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	default:
		ConvTranspose2DTypedScalar(
			format,
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)
	}
}
