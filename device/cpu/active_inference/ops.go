package active_inference

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func FreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	dispatchFreeEnergy(likelihood, posterior, prior, output, count, format)
}

func ExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	if obsCount == 0 {
		return
	}

	dispatchExpectedFreeEnergy(
		predictedObs, preferredObs, predictedState, output,
		obsCount, stateCount, format,
	)
}

func BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	dispatchBeliefUpdate(likelihood, prior, output, count, format)
}

func PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	dispatchPrecisionWeight(errors, precision, output, count, format)
}
