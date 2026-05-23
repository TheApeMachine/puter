//go:build amd64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

//go:noescape
func PrecisionWeightFloat16AVX512Asm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateFloat16AVX512Asm(likelihood, prior, output *uint16, count int)

//go:noescape
func FreeEnergyFloat16AVX512Asm(likelihood, posterior, prior *uint16, count int) uint16

//go:noescape
func ExpectedFreeEnergyFloat16AVX512Asm(
	predictedObs, preferredObs, predictedState *uint16,
	obsCount, stateCount int,
) uint16

func PrecisionWeightFP16AVX512(errors, precision, output []dtype.F16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightFloat16AVX512Asm(
		(*uint16)(&errors[0]),
		(*uint16)(&precision[0]),
		(*uint16)(&output[0]),
		len(errors),
	)
}

func BeliefUpdateFP16AVX512(likelihood, prior, output []dtype.F16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateFloat16AVX512Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&prior[0]),
		(*uint16)(&output[0]),
		len(likelihood),
	)
}

func FreeEnergyFP16AVX512(likelihood, posterior, prior []dtype.F16) dtype.F16 {
	return freeEnergyFloat16F64AMD64(likelihood, posterior, prior)
}

func ExpectedFreeEnergyFP16AVX512(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	return expectedFreeEnergyFloat16F64AMD64(predictedObs, preferredObs, predictedState)
}
