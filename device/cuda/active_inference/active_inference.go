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
	DispatchPrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType)
}
