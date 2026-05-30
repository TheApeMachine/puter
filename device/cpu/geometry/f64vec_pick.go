package geometry

type f64ReduceKernelImpl struct {
	kernel    func(values []float64) float64
	name      string
	available bool
}

type f64DotKernelImpl struct {
	kernel    func(left, right []float64) float64
	name      string
	available bool
}

type f64BinaryKernelImpl struct {
	kernel    func(destination, left, right []float64)
	name      string
	available bool
}

type f64ScaleKernelImpl struct {
	kernel    func(destination, source []float64, scale float64)
	name      string
	available bool
}

type f64AddScalarKernelImpl struct {
	kernel    func(destination, source []float64, offset float64)
	name      string
	available bool
}

type f64UnaryKernelImpl struct {
	kernel    func(destination, source []float64)
	name      string
	available bool
}

func pickF64ReduceKernel(candidates []f64ReduceKernelImpl) func(values []float64) float64 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 reduce kernel available")
}

func pickF64DotKernel(candidates []f64DotKernelImpl) func(left, right []float64) float64 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 dot kernel available")
}

func pickF64BinaryKernel(candidates []f64BinaryKernelImpl) func(destination, left, right []float64) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 binary kernel available")
}

func pickF64ScaleKernel(candidates []f64ScaleKernelImpl) func(destination, source []float64, scale float64) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 scale kernel available")
}

func pickF64AddScalarKernel(candidates []f64AddScalarKernelImpl) func(destination, source []float64, offset float64) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 add-scalar kernel available")
}

func pickF64UnaryKernel(candidates []f64UnaryKernelImpl) func(destination, source []float64) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float64 unary kernel available")
}
