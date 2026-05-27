//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func (host *ComputeHost) DispatchRoPE(
	config device.RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	if seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	host.dispatchError(config.Validate())

	inputShape, err := tensor.NewShape([]int{seqLen, numHeads, headDim})
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)
	inputTensor := host.requireDeviceTensor(input)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteRoPE(
		host.bridge,
		context,
		[]float64{
			config.BaseFreq,
			config.ScalingFactor,
			config.LowFreqFactor,
			config.HighFreqFactor,
		},
		[]int64{
			int64(config.StartPosition),
			int64(config.Mode),
			int64(config.Scaling),
			int64(config.OriginalContext),
		},
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchMultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	host.NotImplemented("MultiAxisRoPE")
}

func (host *ComputeHost) DispatchRoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType,
) {
	headDim := halfDim * 2

	if halfDim == 0 {
		return
	}

	inputShape, err := ShapeFromVector(headDim)
	host.dispatchError(err)

	cosShape, err := ShapeFromVector(halfDim)
	host.dispatchError(err)

	sinShape, err := ShapeFromVector(halfDim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{inputShape, cosShape, sinShape},
		OutputDType: format,
		OutputShape: inputShape,
	}

	inputTensor := host.requireDeviceTensor(input)
	cosTensor := host.requireDeviceTensor(cosBuffer)
	sinTensor := host.requireDeviceTensor(sinBuffer)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"rope_pairs",
		context,
		nil,
		nil,
		[]*DeviceTensor{inputTensor, cosTensor, sinTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchScaledDotProductAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	queryShape, keyShape, valueShape, outputShape, err := attentionShapes(
		seqQ, seqK, depth, valueDim,
	)
	host.dispatchError(err)

	causalParam := int64(0)

	if config.Causal {
		causalParam = 1
	}

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{queryShape, keyShape, valueShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	queryTensor := host.requireDeviceTensor(query)
	keyTensor := host.requireDeviceTensor(key)
	valueTensor := host.requireDeviceTensor(value)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"scaled_dot_product_attention",
		context,
		nil,
		[]int64{causalParam},
		[]*DeviceTensor{queryTensor, keyTensor, valueTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchMultiHeadAttention(
	config device.MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	if seqQ == 0 || seqK == 0 || config.NumHeads == 0 || config.HeadDim == 0 {
		return
	}

	queryShape, err := ShapeFromRowsCols(seqQ, config.NumHeads*config.HeadDim)
	host.dispatchError(err)

	keyShape, err := ShapeFromRowsCols(seqK, kvHeads*config.HeadDim)
	host.dispatchError(err)

	valueShape, err := ShapeFromRowsCols(seqK, kvHeads*config.HeadDim)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(seqQ, config.NumHeads*config.HeadDim)
	host.dispatchError(err)

	causalParam := int64(0)

	if config.Causal {
		causalParam = 1
	}

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{queryShape, keyShape, valueShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	queryTensor := host.requireDeviceTensor(query)
	keyTensor := host.requireDeviceTensor(key)
	valueTensor := host.requireDeviceTensor(value)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"multi_head_attention",
		context,
		[]float64{float64(config.ALiBiSlope)},
		[]int64{
			int64(config.NumHeads),
			int64(config.HeadDim),
			int64(kvHeads),
			causalParam,
			int64(config.WindowSize),
		},
		[]*DeviceTensor{queryTensor, keyTensor, valueTensor},
		outputTensor,
	))
}

func attentionShapes(
	seqQ, seqK, depth, valueDim int,
) (tensor.Shape, tensor.Shape, tensor.Shape, tensor.Shape, error) {
	queryShape, err := ShapeFromRowsCols(seqQ, depth)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	keyShape, err := ShapeFromRowsCols(seqK, depth)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	valueShape, err := ShapeFromRowsCols(seqK, valueDim)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	outputShape, err := ShapeFromRowsCols(seqQ, valueDim)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	return queryShape, keyShape, valueShape, outputShape, nil
}
