//go:build arm64

package active_inference

import (
	"math"

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
		clampedLike := math.Max(activeInferenceEps, float64(loadBF16(likelihood[index])))
		clampedPosterior := math.Max(activeInferenceEps, float64(loadBF16(posteriorValue)))
		clampedPrior := math.Max(activeInferenceEps, float64(loadBF16(prior[index])))
		posteriorFloat := float64(loadBF16(posteriorValue))

		crossEntropy += -posteriorFloat * math.Log(clampedLike)
		kl += posteriorFloat * (math.Log(clampedPosterior) - math.Log(clampedPrior))
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
		predictedClamped := math.Max(activeInferenceEps, predictedFloat)
		preferredClamped := math.Max(activeInferenceEps, float64(loadBF16(preferredObs[index])))

		pragmatic += predictedFloat * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	for _, stateValue := range predictedState {
		stateFloat := float64(loadBF16(stateValue))
		clamped := math.Max(activeInferenceEps, stateFloat)
		epistemic += -stateFloat * math.Log(clamped)
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
		clampedLike := math.Max(activeInferenceEps, float64(loadF16(likelihood[index])))
		clampedPosterior := math.Max(activeInferenceEps, float64(loadF16(posteriorValue)))
		clampedPrior := math.Max(activeInferenceEps, float64(loadF16(prior[index])))
		posteriorFloat := float64(loadF16(posteriorValue))

		crossEntropy += -posteriorFloat * math.Log(clampedLike)
		kl += posteriorFloat * (math.Log(clampedPosterior) - math.Log(clampedPrior))
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
		predictedClamped := math.Max(activeInferenceEps, predictedFloat)
		preferredClamped := math.Max(activeInferenceEps, float64(loadF16(preferredObs[index])))

		pragmatic += predictedFloat * (math.Log(predictedClamped) - math.Log(preferredClamped))
	}

	for _, stateValue := range predictedState {
		stateFloat := float64(loadF16(stateValue))
		clamped := math.Max(activeInferenceEps, stateFloat)
		epistemic += -stateFloat * math.Log(clamped)
	}

	return storeF16(float32(pragmatic + epistemic))
}
