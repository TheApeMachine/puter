//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func (host *ComputeHost) DispatchMaxPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	host.dispatchPool("max_pool2d", config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (host *ComputeHost) DispatchAvgPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	host.dispatchPool("avg_pool2d", config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (host *ComputeHost) DispatchAdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	host.dispatchPool("adaptive_max_pool2d", device.PoolConfig{}, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (host *ComputeHost) DispatchAdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	host.dispatchPool("adaptive_avg_pool2d", device.PoolConfig{}, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (host *ComputeHost) dispatchPool(
	operationName string,
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromNCHW(batch, channels, inHeight, inWidth)
	host.dispatchError(err)

	outputShape, err := ShapeFromNCHW(batch, channels, outHeight, outWidth)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	intParams := []int64{
		int64(config.KernelH),
		int64(config.KernelW),
		int64(config.StrideH),
		int64(config.StrideW),
		int64(config.PaddingH),
		int64(config.PaddingW),
	}

	inputTensor := host.requireDeviceTensor(input)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecutePool(
		host.bridge,
		operationName,
		context,
		intParams,
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) LaunchLayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	host.dispatchLayernorm("layer_norm", input, scale, bias, output, rows, lastDim, format)
}

func (host *ComputeHost) LaunchRMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 || host.bridge == nil {
		return
	}

	host.dispatchError(config.Validate())

	inputShape, err := ShapeFromRowsCols(rows, lastDim)
	host.dispatchError(err)

	scaleShape, err := ShapeFromVector(lastDim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{inputShape, scaleShape},
		OutputDType: format,
		OutputShape: inputShape,
	}

	inputTensor := host.requireDeviceTensor(input)
	scaleTensor := host.requireDeviceTensor(scale)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"rms_norm",
		context,
		[]float64{config.Epsilon},
		nil,
		[]*DeviceTensor{inputTensor, scaleTensor},
		outputTensor,
	))
}

func (host *ComputeHost) LaunchModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	host.NotImplemented("ModulatedLayerNorm")
}

func (host *ComputeHost) dispatchLayernorm(
	operationName string,
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromRowsCols(rows, lastDim)
	host.dispatchError(err)

	vectorShape, err := ShapeFromVector(lastDim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: inputShape,
	}

	inputTensor := host.requireDeviceTensor(input)
	scaleTensor := host.requireDeviceTensor(scale)
	biasTensor := host.requireDeviceTensor(bias)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		operationName,
		context,
		nil,
		nil,
		[]*DeviceTensor{inputTensor, scaleTensor, biasTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchBatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	host.dispatchNormalization(
		"batch_norm_eval",
		[]unsafe.Pointer{input, scale, bias, mean, variance},
		output,
		batch, channels, spatial,
		format,
		nil,
	)
}

func (host *ComputeHost) DispatchInstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	host.dispatchNormalization(
		"instance_norm",
		[]unsafe.Pointer{input, scale, bias},
		output,
		batch, channels, spatial,
		format,
		nil,
	)
}

func (host *ComputeHost) DispatchGroupNorm(
	config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	host.dispatchNormalization(
		"group_norm",
		[]unsafe.Pointer{input, scale, bias},
		output,
		batch, channels, spatial,
		format,
		[]int64{int64(config.Groups)},
	)
}

func (host *ComputeHost) dispatchNormalization(
	operationName string,
	inputs []unsafe.Pointer,
	output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
	intParams []int64,
) {
	if batch == 0 || channels == 0 || spatial == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromBCS(batch, channels, spatial)
	host.dispatchError(err)

	channelShape, err := ShapeFromVector(channels)
	host.dispatchError(err)

	inputDTypes := make([]dtype.DType, len(inputs))
	inputShapes := make([]tensor.Shape, len(inputs))
	deviceTensors := make([]*DeviceTensor, len(inputs))

	for inputIndex, inputPointer := range inputs {
		inputDTypes[inputIndex] = format

		if inputIndex == 0 {
			inputShapes[inputIndex] = inputShape
		}

		if inputIndex > 0 {
			inputShapes[inputIndex] = channelShape
		}

		deviceTensors[inputIndex] = host.requireDeviceTensor(inputPointer)
	}

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: inputDTypes,
		InputShapes: inputShapes,
		OutputDType: format,
		OutputShape: inputShape,
	}

	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		operationName,
		context,
		nil,
		intParams,
		deviceTensors,
		outputTensor,
	))
}
