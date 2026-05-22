//go:build arm64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

var (
	freeEnergyBF16Funcs = []bf16FreeEnergyKernelImpl{
		{FreeEnergyBF16NEON, "neon", true},
		{FreeEnergyBFloat16Generic, "generic", true},
	}
	expectedFreeEnergyBF16Funcs = []bf16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyBF16NEON, "neon", true},
		{ExpectedFreeEnergyBFloat16Generic, "generic", true},
	}
	beliefUpdateBF16Funcs = []bf16BeliefUpdateKernelImpl{
		{BeliefUpdateBF16NEON, "neon", true},
		{BeliefUpdateBFloat16Generic, "generic", true},
	}
	precisionWeightBF16Funcs = []bf16PrecisionWeightKernelImpl{
		{PrecisionWeightBF16NEON, "neon", true},
		{PrecisionWeightBFloat16Generic, "generic", true},
	}

	freeEnergyFP16Funcs = []fp16FreeEnergyKernelImpl{
		{FreeEnergyFP16NEON, "neon", true},
		{FreeEnergyFloat16Generic, "generic", true},
	}
	expectedFreeEnergyFP16Funcs = []fp16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyFP16NEON, "neon", true},
		{ExpectedFreeEnergyFloat16Generic, "generic", true},
	}
	beliefUpdateFP16Funcs = []fp16BeliefUpdateKernelImpl{
		{BeliefUpdateFP16NEON, "neon", true},
		{BeliefUpdateFloat16Generic, "generic", true},
	}
	precisionWeightFP16Funcs = []fp16PrecisionWeightKernelImpl{
		{PrecisionWeightFP16NEON, "neon", true},
		{PrecisionWeightFloat16Generic, "generic", true},
	}
)

func FreeEnergyBF16NEON(likelihood, posterior, prior []dtype.BF16) dtype.BF16 {
	if len(likelihood) == 0 {
		return 0
	}

	return FreeEnergyBFloat16NEONAsm(
		&likelihood[0], &posterior[0], &prior[0], len(likelihood),
	)
}

func ExpectedFreeEnergyBF16NEON(
	predictedObs, preferredObs, predictedState []dtype.BF16,
) dtype.BF16 {
	if len(predictedObs) == 0 {
		return 0
	}

	return ExpectedFreeEnergyBFloat16NEONAsm(
		&predictedObs[0], &preferredObs[0], &predictedState[0],
		len(predictedObs), len(predictedState),
	)
}

func BeliefUpdateBF16NEON(likelihood, prior, output []dtype.BF16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateBFloat16NEONAsm(
		&likelihood[0], &prior[0], &output[0], len(likelihood),
	)
}

func PrecisionWeightBF16NEON(errors, precision, output []dtype.BF16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightBFloat16NEONAsm(
		&errors[0], &precision[0], &output[0], len(errors),
	)
}

func FreeEnergyFP16NEON(likelihood, posterior, prior []dtype.F16) dtype.F16 {
	if len(likelihood) == 0 {
		return 0
	}

	return FreeEnergyFloat16NEONAsm(
		&likelihood[0], &posterior[0], &prior[0], len(likelihood),
	)
}

func ExpectedFreeEnergyFP16NEON(
	predictedObs, preferredObs, predictedState []dtype.F16,
) dtype.F16 {
	if len(predictedObs) == 0 {
		return 0
	}

	return ExpectedFreeEnergyFloat16NEONAsm(
		&predictedObs[0], &preferredObs[0], &predictedState[0],
		len(predictedObs), len(predictedState),
	)
}

func BeliefUpdateFP16NEON(likelihood, prior, output []dtype.F16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateFloat16NEONAsm(
		&likelihood[0], &prior[0], &output[0], len(likelihood),
	)
}

func PrecisionWeightFP16NEON(errors, precision, output []dtype.F16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightFloat16NEONAsm(
		&errors[0], &precision[0], &output[0], len(errors),
	)
}
