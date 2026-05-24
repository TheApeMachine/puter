//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func (host *ComputeHost) DispatchConv2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelHeight == 0 || kernelWidth == 0 {
		return
	}

	inputShape, err := ShapeFromNCHW(batch, inChannels, inHeight, inWidth)
	host.dispatchError(err)

	weightShape, err := tensor.NewShape([]int{outChannels, inChannels, kernelHeight, kernelWidth})
	host.dispatchError(err)

	biasShape, err := ShapeFromVector(outChannels)
	host.dispatchError(err)

	outputShape, err := ShapeFromNCHW(batch, outChannels, outHeight, outWidth)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, weightShape, biasShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	intParams := []int64{
		int64(config.StrideH),
		int64(config.StrideW),
		int64(config.PaddingH),
		int64(config.PaddingW),
		int64(config.DilationH),
		int64(config.DilationW),
	}

	inputTensor := host.requireDeviceTensor(input)
	weightTensor := host.requireDeviceTensor(weight)
	biasTensor := host.requireDeviceTensor(bias)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteConvolution(
		host.bridge,
		"conv2d",
		context,
		intParams,
		inputTensor,
		weightTensor,
		biasTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchConv1D(
	config device.Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
) {
	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelLength == 0 {
		return
	}

	inputShape, err := tensor.NewShape([]int{batch, inChannels, inLength})
	host.dispatchError(err)

	weightShape, err := tensor.NewShape([]int{outChannels, inChannels, kernelLength})
	host.dispatchError(err)

	biasShape, err := ShapeFromVector(outChannels)
	host.dispatchError(err)

	outputShape, err := tensor.NewShape([]int{batch, outChannels, outLength})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, weightShape, biasShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	intParams := []int64{
		int64(config.Stride),
		int64(config.Padding),
		int64(config.Dilation),
	}

	inputTensor := host.requireDeviceTensor(input)
	weightTensor := host.requireDeviceTensor(weight)
	biasTensor := host.requireDeviceTensor(bias)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteConvolution(
		host.bridge,
		"conv1d",
		context,
		intParams,
		inputTensor,
		weightTensor,
		biasTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchConv3D(
	config device.Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inDepth, inHeight, inWidth,
	outChannels, kernelDepth, kernelHeight, kernelWidth,
	outDepth, outHeight, outWidth int,
	format dtype.DType,
) {
	if batch == 0 || inChannels == 0 || outChannels == 0 ||
		kernelDepth == 0 || kernelHeight == 0 || kernelWidth == 0 {
		return
	}

	inputShape, err := tensor.NewShape([]int{batch, inChannels, inDepth, inHeight, inWidth})
	host.dispatchError(err)

	weightShape, err := tensor.NewShape([]int{
		outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth,
	})
	host.dispatchError(err)

	biasShape, err := ShapeFromVector(outChannels)
	host.dispatchError(err)

	outputShape, err := tensor.NewShape([]int{
		batch, outChannels, outDepth, outHeight, outWidth,
	})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, weightShape, biasShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	intParams := []int64{
		int64(config.StrideD),
		int64(config.StrideH),
		int64(config.StrideW),
		int64(config.PaddingD),
		int64(config.PaddingH),
		int64(config.PaddingW),
		int64(config.DilationD),
		int64(config.DilationH),
		int64(config.DilationW),
	}

	inputTensor := host.requireDeviceTensor(input)
	weightTensor := host.requireDeviceTensor(weight)
	biasTensor := host.requireDeviceTensor(bias)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteConvolution(
		host.bridge,
		"conv3d",
		context,
		intParams,
		inputTensor,
		weightTensor,
		biasTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchConvTranspose2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelHeight == 0 || kernelWidth == 0 {
		return
	}

	inputShape, err := ShapeFromNCHW(batch, inChannels, inHeight, inWidth)
	host.dispatchError(err)

	weightShape, err := tensor.NewShape([]int{inChannels, outChannels, kernelHeight, kernelWidth})
	host.dispatchError(err)

	biasShape, err := ShapeFromVector(outChannels)
	host.dispatchError(err)

	outputShape, err := ShapeFromNCHW(batch, outChannels, outHeight, outWidth)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, weightShape, biasShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	intParams := []int64{
		int64(config.StrideH),
		int64(config.StrideW),
		int64(config.PaddingH),
		int64(config.PaddingW),
		int64(config.DilationH),
		int64(config.DilationW),
	}

	inputTensor := host.requireDeviceTensor(input)
	weightTensor := host.requireDeviceTensor(weight)
	biasTensor := host.requireDeviceTensor(bias)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteConvolution(
		host.bridge,
		"conv_transpose2d",
		context,
		intParams,
		inputTensor,
		weightTensor,
		biasTensor,
		outputTensor,
	))
}
