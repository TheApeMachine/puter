package active_inference

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type ActiveInference struct {
	host Host
}

func New(host Host) ActiveInference {
	return ActiveInference{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchBeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType)
	DispatchExpectedFreeEnergy(
		predictedObs, preferredObs, predictedState, output unsafe.Pointer,
		obsCount, stateCount int,
		format dtype.DType,
	)
	DispatchFreeEnergy(
		likelihood, posterior, prior, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchMarkovBlanketPartition(
		adjacency, internal, output unsafe.Pointer,
		nodeCount, internalCount int,
		format dtype.DType,
	)
	DispatchMarkovFlowActive(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	DispatchMarkovFlowInternal(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	DispatchMarkovMutualInformation(
		joint, output unsafe.Pointer,
		xCount, yCount int,
		format dtype.DType,
	)
	DispatchPrediction(
		weights, representation, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	DispatchPredictionError(
		observed, predicted, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchPrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType)
}
