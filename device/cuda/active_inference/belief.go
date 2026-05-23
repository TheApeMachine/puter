//go:build cuda

package active_inference

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activeInference *ActiveInference) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	activeInference.host.DispatchMarkovBlanketPartition(adjacency, internal, output, nodeCount, internalCount, format)
}

func (activeInference *ActiveInference) MarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	activeInference.host.DispatchMarkovFlowActive(mutualInformation, partition, output, nodeCount, format)
}

func (activeInference *ActiveInference) MarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	activeInference.host.DispatchMarkovFlowInternal(mutualInformation, partition, output, nodeCount, format)
}

func (activeInference *ActiveInference) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	activeInference.host.DispatchMarkovMutualInformation(joint, output, xCount, yCount, format)
}

func (activeInference *ActiveInference) Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	activeInference.host.DispatchPrediction(weights, representation, output, outDim, inDim, format)
}

func (activeInference *ActiveInference) PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activeInference.host.DispatchPredictionError(observed, predicted, output, count, format)
}

func (activeInference *ActiveInference) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	activeInference.host.DispatchPrecisionWeight(errors, precision, output, count, format)
}
