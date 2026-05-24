package active_inference

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultActiveInference = New()

func BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	defaultActiveInference.BeliefUpdate(likelihood, prior, output, count, format)
}

func ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType) {
	defaultActiveInference.ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output, obsCount, stateCount, format)
}

func FreeEnergy(likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActiveInference.FreeEnergy(likelihood, posterior, prior, output, count, format)
}

func PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	defaultActiveInference.PrecisionWeight(errors, precision, output, count, format)
}
