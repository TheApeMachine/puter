package active_inference

import "github.com/theapemachine/manifesto/dtype"

var (
	freeEnergyBF16Kernel = pickBF16FreeEnergyKernel(freeEnergyBF16Funcs)
	expectedFreeEnergyBF16Kernel = pickBF16ExpectedFreeEnergyKernel(expectedFreeEnergyBF16Funcs)
	beliefUpdateBF16Kernel = pickBF16BeliefUpdateKernel(beliefUpdateBF16Funcs)
	precisionWeightBF16Kernel = pickBF16PrecisionWeightKernel(precisionWeightBF16Funcs)

	freeEnergyFP16Kernel = pickFP16FreeEnergyKernel(freeEnergyFP16Funcs)
	expectedFreeEnergyFP16Kernel = pickFP16ExpectedFreeEnergyKernel(expectedFreeEnergyFP16Funcs)
	beliefUpdateFP16Kernel = pickFP16BeliefUpdateKernel(beliefUpdateFP16Funcs)
	precisionWeightFP16Kernel = pickFP16PrecisionWeightKernel(precisionWeightFP16Funcs)
)

func FreeEnergyBFloat16Native(likelihood, posterior, prior []dtype.BF16) dtype.BF16 {
	return freeEnergyBF16Kernel(likelihood, posterior, prior)
}

func ExpectedFreeEnergyBFloat16Native(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	return expectedFreeEnergyBF16Kernel(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateBFloat16Native(likelihood, prior, output []dtype.BF16) {
	beliefUpdateBF16Kernel(likelihood, prior, output)
}

func PrecisionWeightBFloat16Native(errors, precision, output []dtype.BF16) {
	precisionWeightBF16Kernel(errors, precision, output)
}

func FreeEnergyFloat16Native(likelihood, posterior, prior []dtype.F16) dtype.F16 {
	return freeEnergyFP16Kernel(likelihood, posterior, prior)
}

func ExpectedFreeEnergyFloat16Native(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	return expectedFreeEnergyFP16Kernel(predictedObs, preferredObs, predictedState)
}

func BeliefUpdateFloat16Native(likelihood, prior, output []dtype.F16) {
	beliefUpdateFP16Kernel(likelihood, prior, output)
}

func PrecisionWeightFloat16Native(errors, precision, output []dtype.F16) {
	precisionWeightFP16Kernel(errors, precision, output)
}
