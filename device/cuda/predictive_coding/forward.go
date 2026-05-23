//go:build cuda

package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (predictiveCoding *PredictiveCoding) Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictiveCoding.host.DispatchPrediction(weights, representation, output, outDim, inDim, format)
}

func (predictiveCoding *PredictiveCoding) PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	predictiveCoding.host.DispatchPredictionError(observed, predicted, output, count, format)
}

func (predictiveCoding *PredictiveCoding) UpdateRepresentation(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictiveCoding.host.DispatchUpdateRepresentation(
		config, weights, representation, predictionError, output, outDim, inDim, format,
	)
}
