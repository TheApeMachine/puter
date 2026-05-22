//go:build amd64

package active_inference

import "unsafe"

//go:noescape
func PrecisionWeightFloat32AVX512Asm(errors, precision, output *float32, count int)

//go:noescape
func BeliefUpdateFloat32AVX512Asm(likelihood, prior, output *float32, count int)

//go:noescape
func FreeEnergyFloat32AVX512Asm(likelihood, posterior, prior *float32, count int) float32

//go:noescape
func ExpectedFreeEnergyFloat32AVX512Asm(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32

func PrecisionWeightF32AVX512(errors, precision, output *float32, count int) {
	if count == 0 {
		return
	}

	PrecisionWeightFloat32AVX512Asm(errors, precision, output, count)
}

func BeliefUpdateF32AVX512(likelihood, prior, output *float32, count int) {
	if count == 0 {
		return
	}

	BeliefUpdateFloat32AVX512Asm(likelihood, prior, output, count)
}

func FreeEnergyF32AVX512(likelihood, posterior, prior *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return FreeEnergyFloat32AVX512Asm(likelihood, posterior, prior, count)
}

func ExpectedFreeEnergyF32AVX512(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32 {
	if obsCount == 0 {
		return 0
	}

	return ExpectedFreeEnergyFloat32AVX512Asm(
		predictedObs, preferredObs, predictedState,
		obsCount, stateCount,
	)
}
