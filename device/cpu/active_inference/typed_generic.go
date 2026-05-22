package active_inference

import "github.com/theapemachine/manifesto/dtype"

func FreeEnergyBFloat16Generic(
	likelihood, posterior, prior []dtype.BF16,
) dtype.BF16 {
	return FreeEnergyBFloat16Scalar(likelihood, posterior, prior)
}

func ExpectedFreeEnergyBFloat16Generic(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	return ExpectedFreeEnergyBFloat16Scalar(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateBFloat16Generic(
	likelihood, prior, output []dtype.BF16,
) {
	BeliefUpdateBFloat16Scalar(likelihood, prior, output)
}

func PrecisionWeightBFloat16Generic(
	errors, precision, output []dtype.BF16,
) {
	PrecisionWeightBFloat16Scalar(errors, precision, output)
}

func FreeEnergyFloat16Generic(
	likelihood, posterior, prior []dtype.F16,
) dtype.F16 {
	return FreeEnergyFloat16Scalar(likelihood, posterior, prior)
}

func ExpectedFreeEnergyFloat16Generic(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	return ExpectedFreeEnergyFloat16Scalar(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateFloat16Generic(
	likelihood, prior, output []dtype.F16,
) {
	BeliefUpdateFloat16Scalar(likelihood, prior, output)
}

func PrecisionWeightFloat16Generic(
	errors, precision, output []dtype.F16,
) {
	PrecisionWeightFloat16Scalar(errors, precision, output)
}
