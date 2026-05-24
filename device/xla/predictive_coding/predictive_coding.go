package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
PredictiveCoding implements device.PredictiveCoding for the XLA backend.
*/
type PredictiveCoding struct {
	host Host
}

/*
Host is the XLA dispatch surface predictive_coding operations call into.
*/
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
	NotImplemented(string)
}

/*
New wires a PredictiveCoding receiver to its XLA dispatch host.
*/
func New(host Host) PredictiveCoding {
	return PredictiveCoding{host: host}
}

func (receiver *PredictiveCoding) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *PredictiveCoding) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
