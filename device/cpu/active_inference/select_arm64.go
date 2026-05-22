//go:build arm64

package active_inference

func FreeEnergyFloat32Native(likelihood, posterior, prior []float32) float32 {
	if len(likelihood) == 0 {
		return 0
	}

	return FreeEnergyF32NEON(
		&likelihood[0], &posterior[0], &prior[0], len(likelihood),
	)
}

func ExpectedFreeEnergyFloat32Native(
	predictedObs, preferredObs, predictedState []float32,
) float32 {
	if len(predictedObs) == 0 {
		return 0
	}

	return ExpectedFreeEnergyF32NEON(
		&predictedObs[0], &preferredObs[0], &predictedState[0],
		len(predictedObs), len(predictedState),
	)
}

func BeliefUpdateFloat32Native(likelihood, prior, output []float32) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateF32NEON(&likelihood[0], &prior[0], &output[0], len(likelihood))
}

func PrecisionWeightFloat32Native(errors, precision, output []float32) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightF32NEON(&errors[0], &precision[0], &output[0], len(errors))
}
