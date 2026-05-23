//go:build xla

package active_inference

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

func (activeInference *ActiveInference) BeliefUpdate(
	likelihood, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activeInference.unimplemented("BeliefUpdate")
}

func (activeInference *ActiveInference) ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer, obsCount, stateCount int, format dtype.DType,) {
	activeInference.unimplemented("ExpectedFreeEnergy")
}

func (activeInference *ActiveInference) FreeEnergy(likelihood, posterior, prior, output unsafe.Pointer, count int, format dtype.DType,) {
	activeInference.unimplemented("FreeEnergy")
}

func (activeInference *ActiveInference) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	activeInference.unimplemented("PrecisionWeight")
}

