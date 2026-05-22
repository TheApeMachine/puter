//go:build amd64

package active_inference

//go:noescape
func PrecisionWeightFloat32SSE2Asm(errors, precision, output *float32, count int)

//go:noescape
func BeliefUpdateFloat32SSE2Asm(likelihood, prior, output *float32, count int)

//go:noescape
func FreeEnergyFloat32SSE2Asm(likelihood, posterior, prior *float32, count int) float32

//go:noescape
func ExpectedFreeEnergyFloat32SSE2Asm(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32

func PrecisionWeightF32SSE2(errors, precision, output *float32, count int) {
	if count == 0 {
		return
	}

	PrecisionWeightFloat32SSE2Asm(errors, precision, output, count)
}

func BeliefUpdateF32SSE2(likelihood, prior, output *float32, count int) {
	if count == 0 {
		return
	}

	BeliefUpdateFloat32SSE2Asm(likelihood, prior, output, count)
}

func FreeEnergyF32SSE2(likelihood, posterior, prior *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return FreeEnergyFloat32SSE2Asm(likelihood, posterior, prior, count)
}

func ExpectedFreeEnergyF32SSE2(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32 {
	if obsCount == 0 {
		return 0
	}

	return ExpectedFreeEnergyFloat32SSE2Asm(
		predictedObs, preferredObs, predictedState,
		obsCount, stateCount,
	)
}
