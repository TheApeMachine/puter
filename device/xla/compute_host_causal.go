//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (host *ComputeHost) DispatchCATE(
	treated, control, output unsafe.Pointer,
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

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"cate",
		context,
		nil,
		nil,
		[]*DeviceTensor{
			host.requireDeviceTensor(treated),
			host.requireDeviceTensor(control),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchCounterfactual(
	observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: vectorShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"counterfactual",
		context,
		[]float64{float64(slope)},
		nil,
		[]*DeviceTensor{
			host.requireDeviceTensor(observedY),
			host.requireDeviceTensor(observedX),
			host.requireDeviceTensor(counterfactualX),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchBackdoorAdjustment(
	conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType,
) {
	if xCount == 0 || zCount == 0 || yCount == 0 || host.bridge == nil {
		return
	}

	conditionalShape, err := tensor.NewShape([]int{xCount, zCount, yCount})
	host.dispatchError(err)

	marginalShape, err := ShapeFromCount(zCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(xCount, yCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{conditionalShape, marginalShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"backdoor_adjustment",
		context,
		nil,
		[]int64{int64(xCount), int64(zCount), int64(yCount)},
		[]*DeviceTensor{
			host.requireDeviceTensor(conditional),
			host.requireDeviceTensor(marginalZ),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchFrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType,
) {
	if xCount == 0 || mediatorCount == 0 || yCount == 0 || host.bridge == nil {
		return
	}

	mediatorShape, err := ShapeFromRowsCols(xCount, mediatorCount)
	host.dispatchError(err)

	outcomeShape, err := tensor.NewShape([]int{xCount, mediatorCount, yCount})
	host.dispatchError(err)

	marginalShape, err := ShapeFromCount(xCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromRowsCols(xCount, yCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{mediatorShape, outcomeShape, marginalShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"frontdoor_adjustment",
		context,
		nil,
		[]int64{int64(xCount), int64(mediatorCount), int64(yCount)},
		[]*DeviceTensor{
			host.requireDeviceTensor(mediatorGivenX),
			host.requireDeviceTensor(outcomeGivenXM),
			host.requireDeviceTensor(marginalX),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchDoIntervene(
	adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType,
) {
	if nodeCount == 0 || host.bridge == nil {
		return
	}

	matrixShape, err := ShapeFromRowsCols(nodeCount, nodeCount)
	host.dispatchError(err)

	intervenedShape, err := ShapeFromCount(intervenedCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, dtype.Int32},
		InputShapes: []tensor.Shape{matrixShape, intervenedShape},
		OutputDType: format,
		OutputShape: matrixShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"do_intervene",
		context,
		nil,
		[]int64{int64(nodeCount), int64(intervenedCount)},
		[]*DeviceTensor{
			host.requireDeviceTensor(adjacency),
			host.requireDeviceTensor(intervened),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchIVEstimate(
	instrument, treatment, outcome, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count < 2 || host.bridge == nil {
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

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"iv_estimate",
		context,
		nil,
		nil,
		[]*DeviceTensor{
			host.requireDeviceTensor(instrument),
			host.requireDeviceTensor(treatment),
			host.requireDeviceTensor(outcome),
		},
		outputTensor,
	))
}

func (host *ComputeHost) DispatchDAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	if conditionalCount == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(conditionalCount)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{vectorShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"dag_markov_factorization",
		context,
		nil,
		nil,
		host.requireDeviceTensor(conditionals),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchMarkovFlow(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	targetLabel int32,
	format dtype.DType,
) {
	if nodeCount == 0 || host.bridge == nil {
		return
	}

	matrixShape, err := ShapeFromRowsCols(nodeCount, nodeCount)
	host.dispatchError(err)

	partitionShape, err := ShapeFromCount(nodeCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, dtype.Int32},
		InputShapes: []tensor.Shape{matrixShape, partitionShape},
		OutputDType: format,
		OutputShape: partitionShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"markov_flow",
		context,
		nil,
		[]int64{int64(nodeCount), int64(targetLabel)},
		[]*DeviceTensor{
			host.requireDeviceTensor(mutualInformation),
			host.requireDeviceTensor(partition),
		},
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchMarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	host.DispatchMarkovFlow(mutualInformation, partition, output, nodeCount, 2, format)
}

func (host *ComputeHost) DispatchMarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	host.DispatchMarkovFlow(mutualInformation, partition, output, nodeCount, 0, format)
}

func (host *ComputeHost) DispatchCholesky(
	input, output unsafe.Pointer,
	matrixOrder int,
	format dtype.DType,
) {
	if matrixOrder == 0 || host.bridge == nil {
		return
	}

	matrixShape, err := ShapeFromRowsCols(matrixOrder, matrixOrder)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{matrixShape},
		OutputDType: format,
		OutputShape: matrixShape,
	}

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"cholesky",
		context,
		nil,
		nil,
		host.requireDeviceTensor(input),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchMarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	if xCount == 0 || yCount == 0 || host.bridge == nil {
		return
	}

	jointShape, err := ShapeFromRowsCols(xCount, yCount)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{jointShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"markov_mutual_information",
		context,
		nil,
		[]int64{int64(xCount), int64(yCount)},
		host.requireDeviceTensor(joint),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchMarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	if nodeCount == 0 || host.bridge == nil {
		return
	}

	adjShape, err := ShapeFromRowsCols(nodeCount, nodeCount)
	host.dispatchError(err)

	internalShape, err := ShapeFromCount(internalCount)
	host.dispatchError(err)

	outputShape, err := ShapeFromCount(nodeCount)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{dtype.Float32, dtype.Int32},
		InputShapes: []tensor.Shape{adjShape, internalShape},
		OutputDType: dtype.Int32,
		OutputShape: outputShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"markov_blanket_partition",
		context,
		nil,
		nil,
		[]*DeviceTensor{
			host.requireDeviceTensor(adjacency),
			host.requireDeviceTensor(internal),
		},
		host.requireDeviceTensor(output),
	))
}
