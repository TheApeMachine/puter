package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

type PredictiveCoding struct {
	host Host
}

func New(host Host) PredictiveCoding {
	return PredictiveCoding{host: host}
}

type Host interface {
	NeedsPlatform()
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
	DispatchUpdateRepresentation(
		config device.PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	DispatchUpdateWeights(
		config device.PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
}
