package active_inference

import "github.com/theapemachine/manifesto/dtype"

type bf16FreeEnergyKernelImpl struct {
	kernel    func(likelihood, posterior, prior []dtype.BF16) dtype.BF16
	name      string
	available bool
}

type bf16ExpectedFreeEnergyKernelImpl struct {
	kernel func(predictedObs, preferredObs, predictedState []dtype.BF16) dtype.BF16
	name   string
	available bool
}

type bf16BeliefUpdateKernelImpl struct {
	kernel    func(likelihood, prior, output []dtype.BF16)
	name      string
	available bool
}

type bf16PrecisionWeightKernelImpl struct {
	kernel    func(errors, precision, output []dtype.BF16)
	name      string
	available bool
}

type fp16FreeEnergyKernelImpl struct {
	kernel    func(likelihood, posterior, prior []dtype.F16) dtype.F16
	name      string
	available bool
}

type fp16ExpectedFreeEnergyKernelImpl struct {
	kernel func(predictedObs, preferredObs, predictedState []dtype.F16) dtype.F16
	name   string
	available bool
}

type fp16BeliefUpdateKernelImpl struct {
	kernel    func(likelihood, prior, output []dtype.F16)
	name      string
	available bool
}

type fp16PrecisionWeightKernelImpl struct {
	kernel    func(errors, precision, output []dtype.F16)
	name      string
	available bool
}

func pickBF16FreeEnergyKernel(candidates []bf16FreeEnergyKernelImpl) func(likelihood, posterior, prior []dtype.BF16) dtype.BF16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no bfloat16 free-energy kernel available")
}

func pickBF16ExpectedFreeEnergyKernel(
	candidates []bf16ExpectedFreeEnergyKernelImpl,
) func(predictedObs, preferredObs, predictedState []dtype.BF16) dtype.BF16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no bfloat16 expected-free-energy kernel available")
}

func pickBF16BeliefUpdateKernel(
	candidates []bf16BeliefUpdateKernelImpl,
) func(likelihood, prior, output []dtype.BF16) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no bfloat16 belief-update kernel available")
}

func pickBF16PrecisionWeightKernel(
	candidates []bf16PrecisionWeightKernelImpl,
) func(errors, precision, output []dtype.BF16) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no bfloat16 precision-weight kernel available")
}

func pickFP16FreeEnergyKernel(candidates []fp16FreeEnergyKernelImpl) func(likelihood, posterior, prior []dtype.F16) dtype.F16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no float16 free-energy kernel available")
}

func pickFP16ExpectedFreeEnergyKernel(
	candidates []fp16ExpectedFreeEnergyKernelImpl,
) func(predictedObs, preferredObs, predictedState []dtype.F16) dtype.F16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no float16 expected-free-energy kernel available")
}

func pickFP16BeliefUpdateKernel(
	candidates []fp16BeliefUpdateKernelImpl,
) func(likelihood, prior, output []dtype.F16) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no float16 belief-update kernel available")
}

func pickFP16PrecisionWeightKernel(
	candidates []fp16PrecisionWeightKernelImpl,
) func(errors, precision, output []dtype.F16) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("active_inference: no float16 precision-weight kernel available")
}
