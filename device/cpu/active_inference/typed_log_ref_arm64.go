//go:build arm64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

func widenF16ToF32(values []dtype.F16) []float32 {
	widened := make([]float32, len(values))

	for index, value := range values {
		widened[index] = loadF16(value)
	}

	return widened
}

func widenBF16ToF32(values []dtype.BF16) []float32 {
	widened := make([]float32, len(values))

	for index, value := range values {
		widened[index] = loadBF16(value)
	}

	return widened
}

func FreeEnergyFP16F32LogRef(
	likelihood, posterior, prior []dtype.F16,
) dtype.F16 {
	if len(likelihood) == 0 {
		return 0
	}

	likelihoodF32 := widenF16ToF32(likelihood)
	posteriorF32 := widenF16ToF32(posterior)
	priorF32 := widenF16ToF32(prior)

	return storeF16(FreeEnergyF32NEON(
		&likelihoodF32[0], &posteriorF32[0], &priorF32[0], len(likelihood),
	))
}

func ExpectedFreeEnergyFP16F32LogRef(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	if len(predictedObs) == 0 {
		return 0
	}

	predictedObsF32 := widenF16ToF32(predictedObs)
	preferredObsF32 := widenF16ToF32(preferredObs)
	predictedStateF32 := widenF16ToF32(predictedState)

	return storeF16(ExpectedFreeEnergyF32NEON(
		&predictedObsF32[0], &preferredObsF32[0], &predictedStateF32[0],
		len(predictedObs), len(predictedState),
	))
}

func FreeEnergyBF16F32LogRef(
	likelihood, posterior, prior []dtype.BF16,
) dtype.BF16 {
	if len(likelihood) == 0 {
		return 0
	}

	likelihoodF32 := widenBF16ToF32(likelihood)
	posteriorF32 := widenBF16ToF32(posterior)
	priorF32 := widenBF16ToF32(prior)

	return storeBF16(FreeEnergyF32NEON(
		&likelihoodF32[0], &posteriorF32[0], &priorF32[0], len(likelihood),
	))
}

func ExpectedFreeEnergyBF16F32LogRef(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	if len(predictedObs) == 0 {
		return 0
	}

	predictedObsF32 := widenBF16ToF32(predictedObs)
	preferredObsF32 := widenBF16ToF32(preferredObs)
	predictedStateF32 := widenBF16ToF32(predictedState)

	return storeBF16(ExpectedFreeEnergyF32NEON(
		&predictedObsF32[0], &preferredObsF32[0], &predictedStateF32[0],
		len(predictedObs), len(predictedState),
	))
}
