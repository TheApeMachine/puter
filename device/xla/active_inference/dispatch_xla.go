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
	activeInference.host.DispatchBeliefUpdate(likelihood, prior, output, count, format)
}

func (activeInference *ActiveInference) ExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	activeInference.host.DispatchExpectedFreeEnergy(
		predictedObs, preferredObs, predictedState, output,
		obsCount, stateCount, format,
	)
}

func (activeInference *ActiveInference) FreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activeInference.host.DispatchFreeEnergy(likelihood, posterior, prior, output, count, format)
}

func (activeInference *ActiveInference) PrecisionWeight(
	errors, precision, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activeInference.host.DispatchPrecisionWeight(errors, precision, output, count, format)
}
