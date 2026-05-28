//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/xla/elementwise"
)

func (host *ComputeHost) DispatchQuant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{srcFormat},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: dstFormat,
		OutputShape: inputShape,
	}

	inputTensor := host.requireDeviceTensor(src)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteConvertUnary(
		host.bridge,
		"quant_int8",
		context,
		[]float64{float64(config.Scale)},
		[]int64{int64(config.ZeroPoint)},
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchDequant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{srcFormat},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: dstFormat,
		OutputShape: inputShape,
	}

	inputTensor := host.requireDeviceTensor(src)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteConvertUnary(
		host.bridge,
		"dequant_int8",
		context,
		[]float64{float64(config.Scale)},
		[]int64{int64(config.ZeroPoint)},
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchDequant4(
	dst, src unsafe.Pointer,
	elementCount int,
	config device.DequantInt4Config,
	dstFormat, srcFormat dtype.DType,
) {
	if elementCount == 0 || host.bridge == nil {
		return
	}

	packedCount := (elementCount + 1) / 2
	inputShape, err := ShapeFromCount(packedCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromCount(elementCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{srcFormat},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: dstFormat,
		OutputShape: outputShape,
	}

	inputTensor := host.requireDeviceTensor(src)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteConvertUnary(
		host.bridge,
		"dequant_int4",
		context,
		[]float64{float64(config.Scale)},
		[]int64{int64(config.ZeroPoint)},
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchBind(
	left, right, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.BinaryElementwise(output, left, right, format, elementwise.BinaryMul)
}

func (host *ComputeHost) DispatchBundle(
	left, right, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.BinaryElementwise(output, left, right, format, elementwise.BinaryAdd)
}

func (host *ComputeHost) DispatchSimilarity(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.DotProduct(dst, left, right, count, format)
}

func (host *ComputeHost) DispatchPermute(
	config device.VSAConfig,
	input, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	shift := config.Shift % count

	if shift < 0 {
		shift += count
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)
	inputTensor := host.requireDeviceTensor(input)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"cyclic_permute",
		context,
		nil,
		[]int64{int64(shift)},
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchInversePermute(
	config device.VSAConfig,
	input, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	inverted := config
	inverted.Shift = -config.Shift
	host.DispatchPermute(inverted, input, output, count, format)
}

func (host *ComputeHost) DispatchBeliefUpdate(
	likelihood, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: vectorShape,
	}

	likelihoodTensor := host.requireDeviceTensor(likelihood)
	priorTensor := host.requireDeviceTensor(prior)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"belief_update",
		context,
		nil,
		nil,
		[]*DeviceTensor{likelihoodTensor, priorTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchPrecisionWeight(
	errors, precision, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: vectorShape,
	}

	errorsTensor := host.requireDeviceTensor(errors)
	precisionTensor := host.requireDeviceTensor(precision)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"precision_weight",
		context,
		nil,
		nil,
		[]*DeviceTensor{errorsTensor, precisionTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchFreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.requireDeviceTensor(output)

	likelihoodTensor := host.requireDeviceTensor(likelihood)
	posteriorTensor := host.requireDeviceTensor(posterior)
	priorTensor := host.requireDeviceTensor(prior)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"free_energy",
		context,
		nil,
		nil,
		[]*DeviceTensor{likelihoodTensor, posteriorTensor, priorTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	if obsCount == 0 || host.bridge == nil {
		return
	}

	obsShape, err := ShapeFromCount(obsCount)
	host.dispatchError(err)

	stateShape, err := ShapeFromCount(stateCount)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{obsShape, obsShape, stateShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.requireDeviceTensor(output)

	predictedObsTensor := host.requireDeviceTensor(predictedObs)
	preferredObsTensor := host.requireDeviceTensor(preferredObs)
	predictedStateTensor := host.requireDeviceTensor(predictedState)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"expected_free_energy",
		context,
		nil,
		nil,
		[]*DeviceTensor{predictedObsTensor, preferredObsTensor, predictedStateTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchPrediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	host.MatmulLaunch(output, weights, representation, outDim, inDim, 1, format)
}

func (host *ComputeHost) DispatchPredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.BinaryElementwise(output, observed, predicted, format, elementwise.BinarySub)
}

func (host *ComputeHost) DispatchUpdateRepresentation(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	if outDim == 0 || inDim == 0 || host.bridge == nil {
		return
	}

	weightShape, repShape, errShape, err := researchMatrixVectorShapes(outDim, inDim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{weightShape, repShape, errShape},
		OutputDType: format,
		OutputShape: repShape,
	}

	weightsTensor := host.requireDeviceTensor(weights)
	representationTensor := host.requireDeviceTensor(representation)
	errorTensor := host.requireDeviceTensor(predictionError)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"update_representation",
		context,
		[]float64{float64(config.LearningRate)},
		nil,
		[]*DeviceTensor{weightsTensor, representationTensor, errorTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchUpdateWeights(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	if outDim == 0 || inDim == 0 || host.bridge == nil {
		return
	}

	weightShape, repShape, errShape, err := researchMatrixVectorShapes(outDim, inDim)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{weightShape, repShape, errShape},
		OutputDType: format,
		OutputShape: weightShape,
	}

	weightsTensor := host.requireDeviceTensor(weights)
	representationTensor := host.requireDeviceTensor(representation)
	errorTensor := host.requireDeviceTensor(predictionError)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"update_weights",
		context,
		[]float64{float64(config.LearningRate)},
		nil,
		[]*DeviceTensor{weightsTensor, representationTensor, errorTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchHawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
) {
	if queryCount == 0 || host.bridge == nil {
		return
	}

	eventShape, err := ShapeFromCount(eventCount)
	host.dispatchError(err)

	queryShape, err := ShapeFromCount(queryCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{eventShape, queryShape},
		OutputDType: format,
		OutputShape: queryShape,
	}

	eventTensor := host.requireDeviceTensor(eventTimes)
	queryTensor := host.requireDeviceTensor(queryTimes)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"hawkes_intensity",
		context,
		[]float64{float64(mu), float64(alpha), float64(beta)},
		nil,
		[]*DeviceTensor{eventTensor, queryTensor},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchHawkesKernelMatrix(
	eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType,
) {
	if eventCount == 0 || host.bridge == nil {
		return
	}

	eventShape, err := ShapeFromCount(eventCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(eventCount, eventCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{eventShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	eventTensor := host.requireDeviceTensor(eventTimes)
	outputTensor := host.requireDeviceTensor(output)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"hawkes_kernel_matrix",
		context,
		[]float64{float64(alpha), float64(beta)},
		nil,
		eventTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DispatchHawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
) {
	if eventCount == 0 || host.bridge == nil {
		return
	}

	eventShape, err := ShapeFromCount(eventCount)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{eventShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.requireDeviceTensor(output)

	eventTensor := host.requireDeviceTensor(eventTimes)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"hawkes_log_likelihood",
		context,
		[]float64{float64(totalT), float64(mu), float64(alpha), float64(beta)},
		nil,
		eventTensor,
		outputTensor,
	))
}

func researchMatrixVectorShapes(outDim, inDim int) (tensor.Shape, tensor.Shape, tensor.Shape, error) {
	weightShape, err := ShapeFromRowsCols(outDim, inDim)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	repShape, err := ShapeFromCount(inDim)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	errShape, err := ShapeFromCount(outDim)

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	return weightShape, repShape, errShape, nil
}
