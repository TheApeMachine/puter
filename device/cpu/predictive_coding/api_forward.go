package predictive_coding

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultPredictiveCoding = New()

func Prediction(weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	defaultPredictiveCoding.Prediction(weights, representation, output, outDim, inDim, format)
}

func PredictionError(observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultPredictiveCoding.PredictionError(observed, predicted, output, count, format)
}

func UpdateRepresentation(config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	defaultPredictiveCoding.UpdateRepresentation(config, weights, representation, predictionError, output, outDim, inDim, format)
}

func UpdateWeights(config PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	defaultPredictiveCoding.UpdateWeights(config, weights, representation, predictionError, output, outDim, inDim, format)
}
