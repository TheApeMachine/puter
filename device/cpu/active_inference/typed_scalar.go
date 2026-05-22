package active_inference

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

func loadBF16(value dtype.BF16) float32 {
	return (&value).Float32()
}

func storeBF16(value float32) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(value)
}

func loadF16(value dtype.F16) float32 {
	return value.Float32()
}

func storeF16(value float32) dtype.F16 {
	return dtype.Fromfloat32(value)
}

func FreeEnergyBFloat16Scalar(likelihood, posterior, prior []dtype.BF16) dtype.BF16 {
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

func ExpectedFreeEnergyBFloat16Scalar(
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

func BeliefUpdateBFloat16Scalar(likelihood, prior, output []dtype.BF16) {
	if len(likelihood) == 0 {
		return
	}

	var sum float64

	for index, likeValue := range likelihood {
		product := loadBF16(likeValue) * loadBF16(prior[index])
		output[index] = storeBF16(product)
		sum += float64(product)
	}

	if sum == 0 {
		return
	}

	normalizer := float32(1.0 / sum)

	for index := range output {
		output[index] = storeBF16(loadBF16(output[index]) * normalizer)
	}
}

func PrecisionWeightBFloat16Scalar(errors, precision, output []dtype.BF16) {
	for index, value := range errors {
		output[index] = storeBF16(loadBF16(value) * loadBF16(precision[index]))
	}
}

func FreeEnergyFloat16Scalar(likelihood, posterior, prior []dtype.F16) dtype.F16 {
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

func ExpectedFreeEnergyFloat16Scalar(
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

func BeliefUpdateFloat16Scalar(likelihood, prior, output []dtype.F16) {
	if len(likelihood) == 0 {
		return
	}

	var sum float64

	for index, likeValue := range likelihood {
		product := loadF16(likeValue) * loadF16(prior[index])
		output[index] = storeF16(product)
		sum += float64(product)
	}

	if sum == 0 {
		return
	}

	normalizer := float32(1.0 / sum)

	for index := range output {
		output[index] = storeF16(loadF16(output[index]) * normalizer)
	}
}

func PrecisionWeightFloat16Scalar(errors, precision, output []dtype.F16) {
	for index, value := range errors {
		output[index] = storeF16(loadF16(value) * loadF16(precision[index]))
	}
}
