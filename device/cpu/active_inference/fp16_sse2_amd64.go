//go:build amd64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

//go:noescape
func PrecisionWeightFloat16SSE2Asm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateFloat16SSE2Asm(likelihood, prior, output *uint16, count int)

//go:noescape
func FreeEnergyFloat16SSE2Asm(likelihood, posterior, prior *uint16, count int) uint16

//go:noescape
func ExpectedFreeEnergyFloat16SSE2Asm(
	predictedObs, preferredObs, predictedState *uint16,
	obsCount, stateCount int,
) uint16

func PrecisionWeightFP16SSE2(errors, precision, output []dtype.F16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightFloat16SSE2Asm(
		(*uint16)(&errors[0]),
		(*uint16)(&precision[0]),
		(*uint16)(&output[0]),
		len(errors),
	)
}

func BeliefUpdateFP16SSE2(likelihood, prior, output []dtype.F16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateFloat16SSE2Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&prior[0]),
		(*uint16)(&output[0]),
		len(likelihood),
	)
}

func FreeEnergyFP16SSE2(likelihood, posterior, prior []dtype.F16) dtype.F16 {
	if len(likelihood) == 0 {
		return 0
	}

	return dtype.F16(FreeEnergyFloat16SSE2Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&posterior[0]),
		(*uint16)(&prior[0]),
		len(likelihood),
	))
}

func ExpectedFreeEnergyFP16SSE2(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	if len(predictedObs) == 0 {
		return 0
	}

	return dtype.F16(ExpectedFreeEnergyFloat16SSE2Asm(
		(*uint16)(&predictedObs[0]),
		(*uint16)(&preferredObs[0]),
		(*uint16)(&predictedState[0]),
		len(predictedObs), len(predictedState),
	))
}
