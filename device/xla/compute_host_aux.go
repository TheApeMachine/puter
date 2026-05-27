//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/xla/elementwise"
)

func (host *ComputeHost) DispatchApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.BinaryElementwise(output, input, mask, format, elementwise.BinaryAdd)
}

func (host *ComputeHost) DispatchCausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	outputShape, err := ShapeFromRowsCols(seqQ, seqK)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		OutputDType: format,
		OutputShape: outputShape,
	}

	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteNullary(
		host.bridge,
		"causal_mask",
		context,
		[]int64{int64(seqQ), int64(seqK)},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	scoreShape, err := ShapeFromRowsCols(seqQ, seqK)
	host.dispatchError(err)

	slopeShape, err := ShapeFromVector(1)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{scoreShape, slopeShape},
		OutputDType: format,
		OutputShape: scoreShape,
	}

	scoreTensor := host.requireDeviceTensor(scores)
	slopeTensor := host.requireDeviceTensor(slope)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"alibi_bias",
		context,
		nil,
		nil,
		[]*DeviceTensor{scoreTensor, slopeTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchDropout(
	dst, src unsafe.Pointer,
	count int,
	config device.DropoutConfig,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)
	inputTensor := host.requireDeviceTensor(src)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteDropout(
		host.bridge,
		context,
		config.Rate,
		config.Seed,
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchEmbeddingLookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	if vocab == 0 || hidden == 0 || indexCount == 0 {
		return
	}

	tableShape, err := ShapeFromRowsCols(vocab, hidden)
	host.dispatchError(err)

	indicesShape, err := ShapeFromVector(indexCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(indexCount, hidden)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, dtype.Int32},
		InputShapes: []tensor.Shape{tableShape, indicesShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	tableTensor := host.requireDeviceTensor(table)
	indicesTensor := host.requireDeviceTensor(indices)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"embedding_lookup",
		context,
		nil,
		nil,
		[]*DeviceTensor{tableTensor, indicesTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchEmbeddingBag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	if vocab == 0 || hidden == 0 || bagCount == 0 || indexCount == 0 {
		return
	}

	tableShape, err := ShapeFromRowsCols(vocab, hidden)
	host.dispatchError(err)

	indicesShape, err := ShapeFromVector(indexCount)
	host.dispatchError(err)

	offsetsShape, err := ShapeFromVector(bagCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(bagCount, hidden)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, dtype.Int32, dtype.Int32},
		InputShapes: []tensor.Shape{tableShape, indicesShape, offsetsShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	tableTensor := host.requireDeviceTensor(table)
	indicesTensor := host.requireDeviceTensor(indices)
	offsetsTensor := host.requireDeviceTensor(offsets)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"embedding_bag",
		context,
		nil,
		nil,
		[]*DeviceTensor{tableTensor, indicesTensor, offsetsTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchTimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	if count == 0 || dim == 0 {
		return
	}

	host.dispatchError(config.Validate())

	inputShape, err := ShapeFromVector(count)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(count, dim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{dtype.Float32},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	inputTensor := host.requireDeviceTensor(timesteps)
	outputTensor := host.requireDeviceTensor(output)
	flip := int64(0)

	if config.FlipSinToCos {
		flip = 1
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"timestep_embedding",
		context,
		[]float64{
			float64(config.MaxPeriod),
			float64(config.DownscaleFreqShift),
			float64(config.TimestepDivisor),
		},
		[]int64{flip},
		[]*DeviceTensor{inputTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchGreedySample(
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	if vocabSize == 0 || host.bridge == nil {
		return 0
	}

	inputShape, err := ShapeFromVector(vocabSize)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: dtype.Int32,
		OutputShape: scalarShape,
	}

	outputTensor := host.borrowScalarBuffer(dtype.Int32)
	defer outputTensor.Close()

	inputTensor := host.requireDeviceTensor(logits)

	host.dispatchError(host.builder.ExecuteGreedySample(
		host.bridge,
		context,
		inputTensor,
		outputTensor,
	))

	return host.readScalarInt32(outputTensor)
}

func (host *ComputeHost) readScalarInt32(deviceTensor *DeviceTensor) int32 {
	_, bytesOut, err := host.bridge.download(deviceTensor)
	host.dispatchError(err)

	if len(bytesOut) < 4 {
		host.dispatchError(&loweringError{message: "empty XLA int32 scalar download"})
	}

	return int32(bytesOut[0]) |
		int32(bytesOut[1])<<8 |
		int32(bytesOut[2])<<16 |
		int32(bytesOut[3])<<24
}
