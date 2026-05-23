//go:build amd64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

//go:noescape
func PrecisionWeightBFloat16SSE2Asm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateBFloat16SSE2Asm(likelihood, prior, output *uint16, count int)

//go:noescape
func FreeEnergyBFloat16SSE2Asm(likelihood, posterior, prior *uint16, count int) uint16

//go:noescape
func ExpectedFreeEnergyBFloat16SSE2Asm(
	predictedObs, preferredObs, predictedState *uint16,
	obsCount, stateCount int,
) uint16

func PrecisionWeightBF16SSE2(errors, precision, output []dtype.BF16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightBFloat16SSE2Asm(
		(*uint16)(&errors[0]),
		(*uint16)(&precision[0]),
		(*uint16)(&output[0]),
		len(errors),
	)
}

func BeliefUpdateBF16SSE2(likelihood, prior, output []dtype.BF16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateBFloat16SSE2Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&prior[0]),
		(*uint16)(&output[0]),
		len(likelihood),
	)
}

func FreeEnergyBF16SSE2(likelihood, posterior, prior []dtype.BF16) dtype.BF16 {
	if len(likelihood) == 0 {
		return 0
	}

	return dtype.BF16(FreeEnergyBFloat16SSE2Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&posterior[0]),
		(*uint16)(&prior[0]),
		len(likelihood),
	))
}

func ExpectedFreeEnergyBF16SSE2(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	if len(predictedObs) == 0 {
		return 0
	}

	return dtype.BF16(ExpectedFreeEnergyBFloat16SSE2Asm(
		(*uint16)(&predictedObs[0]),
		(*uint16)(&preferredObs[0]),
		(*uint16)(&predictedState[0]),
		len(predictedObs), len(predictedState),
	))
}
