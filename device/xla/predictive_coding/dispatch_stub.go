//go:build !xla

package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (predictiveCoding *PredictiveCoding) Prediction( weights, representation, output unsafe.Pointer, outDim, inDim int, format dtype.DType, ) {
	predictiveCoding.stubHost()
}

func (predictiveCoding *PredictiveCoding) PredictionError( observed, predicted, output unsafe.Pointer, count int, format dtype.DType, ) {
	predictiveCoding.stubHost()
}

func (predictiveCoding *PredictiveCoding) UpdateRepresentation( config PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType, ) {
	predictiveCoding.stubHost()
}

func (predictiveCoding *PredictiveCoding) UpdateWeights( config PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType, ) {
	predictiveCoding.stubHost()
}

