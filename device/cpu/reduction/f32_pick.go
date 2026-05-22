package reduction

type f32ReduceKernelImpl struct {
	kernel    func(values *float32, count int) float32
	name      string
	available bool
}

func pickF32ReduceKernel(
	candidates []f32ReduceKernelImpl,
) func(values *float32, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float32 reduce kernel available")
}

type bf16SumKernelImpl struct {
	kernel    func(values *uint16, count int) uint16
	name      string
	available bool
}

func pickBF16SumKernel(
	candidates []bf16SumKernelImpl,
) func(values *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 sum kernel available")
}

type fp16SumKernelImpl struct {
	kernel    func(values *uint16, count int) uint16
	name      string
	available bool
}

func pickFP16SumKernel(
	candidates []fp16SumKernelImpl,
) func(values *uint16, count int) uint16 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 sum kernel available")
}

type bf16ProdKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickBF16ProdKernel(
	candidates []bf16ProdKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 prod kernel available")
}

type fp16ProdKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickFP16ProdKernel(
	candidates []fp16ProdKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 prod kernel available")
}

type bf16MinKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickBF16MinKernel(
	candidates []bf16MinKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 min kernel available")
}

type bf16MaxKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickBF16MaxKernel(
	candidates []bf16MaxKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 max kernel available")
}

type bf16L1NormKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickBF16L1NormKernel(
	candidates []bf16L1NormKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no bfloat16 l1 norm kernel available")
}

type fp16MinKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickFP16MinKernel(
	candidates []fp16MinKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 min kernel available")
}

type fp16MaxKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickFP16MaxKernel(
	candidates []fp16MaxKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 max kernel available")
}

type fp16L1NormKernelImpl struct {
	kernel    func(values *uint16, count int) float32
	name      string
	available bool
}

func pickFP16L1NormKernel(
	candidates []fp16L1NormKernelImpl,
) func(values *uint16, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("reduction: no float16 l1 norm kernel available")
}
