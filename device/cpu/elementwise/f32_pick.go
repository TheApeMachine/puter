package elementwise

type f32BinaryKernelImpl struct {
	kernel    func(dst, left, right *float32, count int)
	name      string
	available bool
}

type f32UnaryKernelImpl struct {
	kernel    func(dst, src *float32, count int)
	name      string
	available bool
}

type f32AxpyKernelImpl struct {
	kernel    func(y, x *float32, alpha float32, count int)
	name      string
	available bool
}

type f64BinaryKernelImpl struct {
	kernel    func(dst, left, right *float64, count int)
	name      string
	available bool
}

type uint16BinaryKernelImpl struct {
	kernel    func(dst, left, right *uint16, count int)
	name      string
	available bool
}

type uint16UnaryKernelImpl struct {
	kernel    func(dst, src *uint16, count int)
	name      string
	available bool
}

type uint16AxpyKernelImpl struct {
	kernel    func(y, x *uint16, alpha float32, count int)
	name      string
	available bool
}

func pickF32BinaryKernel(
	candidates []f32BinaryKernelImpl,
) func(dst, left, right *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no float32 binary kernel available")
}

func pickF32UnaryKernel(
	candidates []f32UnaryKernelImpl,
) func(dst, src *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no float32 unary kernel available")
}

func pickF32AxpyKernel(
	candidates []f32AxpyKernelImpl,
) func(y, x *float32, alpha float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no float32 axpy kernel available")
}

func pickF64BinaryKernel(
	candidates []f64BinaryKernelImpl,
) func(dst, left, right *float64, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no float64 binary kernel available")
}

func pickUInt16BinaryKernel(
	candidates []uint16BinaryKernelImpl,
) func(dst, left, right *uint16, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no uint16 binary kernel available")
}

func pickUInt16UnaryKernel(
	candidates []uint16UnaryKernelImpl,
) func(dst, src *uint16, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no uint16 unary kernel available")
}

func pickUInt16AxpyKernel(
	candidates []uint16AxpyKernelImpl,
) func(y, x *uint16, alpha float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("elementwise: no uint16 axpy kernel available")
}
