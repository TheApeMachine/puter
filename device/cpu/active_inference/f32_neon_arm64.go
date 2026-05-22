//go:build arm64

package active_inference

//go:noescape
func PrecisionWeightFloat32NEONAsm(errors, precision, output *float32, count int)

//go:noescape
func BeliefUpdateFloat32NEONAsm(likelihood, prior, output *float32, count int)

//go:noescape
func FreeEnergyFloat32NEONAsm(likelihood, posterior, prior *float32, count int) float32

//go:noescape
func ExpectedFreeEnergyFloat32NEONAsm(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32

func PrecisionWeightF32NEON(errors, precision, output *float32, count int) {
	if count == 0 {
		return
	}

	PrecisionWeightFloat32NEONAsm(errors, precision, output, count)
}

func BeliefUpdateF32NEON(likelihood, prior, output *float32, count int) {
	if count == 0 {
		return
	}

	BeliefUpdateFloat32NEONAsm(likelihood, prior, output, count)
}

func FreeEnergyF32NEON(likelihood, posterior, prior *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return FreeEnergyFloat32NEONAsm(likelihood, posterior, prior, count)
}

func ExpectedFreeEnergyF32NEON(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32 {
	if obsCount == 0 {
		return 0
	}

	return ExpectedFreeEnergyFloat32NEONAsm(
		predictedObs, preferredObs, predictedState,
		obsCount, stateCount,
	)
}
