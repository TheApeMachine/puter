package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchConv2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	if batch*inChannels*inHeight*inWidth == 0 ||
		batch*outChannels*outHeight*outWidth == 0 {
		return
	}

	if format == dtype.Float32 {
		runConv2DF32(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	runConv2DReduced(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)
}

func dispatchConv1D(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
) {
	if batch*inChannels*inLength == 0 ||
		batch*outChannels*outLength == 0 {
		return
	}

	if format == dtype.Float32 {
		runConv1DF32(
			config,
			input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)

		return
	}

	runConv1DReduced(
		config,
		input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
	)
}

func dispatchConv3D(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
) {
	if batch*inChannels*inD*inH*inW == 0 ||
		batch*outChannels*outD*outH*outW == 0 {
		return
	}

	if format == dtype.Float32 {
		runConv3DF32(
			config,
			input, weight, bias, output,
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)

		return
	}

	runConv3DReduced(
		config,
		input, weight, bias, output,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		format,
	)
}

func dispatchConvTranspose2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	if batch*inChannels*inHeight*inWidth == 0 ||
		batch*outChannels*outHeight*outWidth == 0 {
		return
	}

	if format == dtype.Float32 {
		runConvTranspose2DF32(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	runConvTranspose2DReduced(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)
}
