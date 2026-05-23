//go:build arm64

package active_inference

import (
	"github.com/theapemachine/manifesto/dtype"
)

func freeEnergyBFloat16F64NEON(
	likelihood, posterior, prior []dtype.BF16,
) dtype.BF16 {
	if len(likelihood) == 0 {
		return 0
	}

	var crossEntropy, kl float64

	for index, posteriorValue := range posterior {
		posteriorFloat := float64(loadBF16(posteriorValue))
		clampedLike := clampActiveInferenceLog(float64(loadBF16(likelihood[index])))
		clampedPosterior := clampActiveInferenceLog(float64(loadBF16(posteriorValue)))
		clampedPrior := clampActiveInferenceLog(float64(loadBF16(prior[index])))

		crossEntropy += -posteriorFloat * clampedLike
		kl += posteriorFloat * (clampedPosterior - clampedPrior)
	}

	return storeBF16(float32(crossEntropy + kl))
}

func expectedFreeEnergyBFloat16F64NEON(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	if len(predictedObs) == 0 {
		return 0
	}

	var pragmatic, epistemic float64

	for index, predicted := range predictedObs {
		predictedFloat := float64(loadBF16(predicted))
		predictedClamped := clampActiveInferenceLog(predictedFloat)
		preferredClamped := clampActiveInferenceLog(float64(loadBF16(preferredObs[index])))

		pragmatic += predictedFloat * (predictedClamped - preferredClamped)
	}

	for _, stateValue := range predictedState {
		stateFloat := float64(loadBF16(stateValue))
		clamped := clampActiveInferenceLog(stateFloat)
		epistemic += -stateFloat * clamped
	}

	return storeBF16(float32(pragmatic + epistemic))
}

func freeEnergyFloat16F64NEON(
	likelihood, posterior, prior []dtype.F16,
) dtype.F16 {
	if len(likelihood) == 0 {
		return 0
	}

	var crossEntropy, kl float64

	for index, posteriorValue := range posterior {
		posteriorFloat := float64(loadF16(posteriorValue))
		clampedLike := clampActiveInferenceLog(float64(loadF16(likelihood[index])))
		clampedPosterior := clampActiveInferenceLog(float64(loadF16(posteriorValue)))
		clampedPrior := clampActiveInferenceLog(float64(loadF16(prior[index])))

		crossEntropy += -posteriorFloat * clampedLike
		kl += posteriorFloat * (clampedPosterior - clampedPrior)
	}

	return storeF16(float32(crossEntropy + kl))
}

func expectedFreeEnergyFloat16F64NEON(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	if len(predictedObs) == 0 {
		return 0
	}

	var pragmatic, epistemic float64

	for index, predicted := range predictedObs {
		predictedFloat := float64(loadF16(predicted))
		predictedClamped := clampActiveInferenceLog(predictedFloat)
		preferredClamped := clampActiveInferenceLog(float64(loadF16(preferredObs[index])))

		pragmatic += predictedFloat * (predictedClamped - preferredClamped)
	}

	for _, stateValue := range predictedState {
		stateFloat := float64(loadF16(stateValue))
		clamped := clampActiveInferenceLog(stateFloat)
		epistemic += -stateFloat * clamped
	}

	return storeF16(float32(pragmatic + epistemic))
}
