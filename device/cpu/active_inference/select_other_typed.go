//go:build !arm64 && !amd64

package active_inference

var (
	freeEnergyBF16Funcs = []bf16FreeEnergyKernelImpl{
		{FreeEnergyBFloat16Generic, "generic", true},
	}
	expectedFreeEnergyBF16Funcs = []bf16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyBFloat16Generic, "generic", true},
	}
	beliefUpdateBF16Funcs = []bf16BeliefUpdateKernelImpl{
		{BeliefUpdateBFloat16Generic, "generic", true},
	}
	precisionWeightBF16Funcs = []bf16PrecisionWeightKernelImpl{
		{PrecisionWeightBFloat16Generic, "generic", true},
	}

	freeEnergyFP16Funcs = []fp16FreeEnergyKernelImpl{
		{FreeEnergyFloat16Generic, "generic", true},
	}
	expectedFreeEnergyFP16Funcs = []fp16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyFloat16Generic, "generic", true},
	}
	beliefUpdateFP16Funcs = []fp16BeliefUpdateKernelImpl{
		{BeliefUpdateFloat16Generic, "generic", true},
	}
	precisionWeightFP16Funcs = []fp16PrecisionWeightKernelImpl{
		{PrecisionWeightFloat16Generic, "generic", true},
	}
)
