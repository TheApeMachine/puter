package geometry

type geometricProductKernelImpl struct {
	kernel    func(left, right, destination *float64)
	name      string
	available bool
}

type rotorSimilarityKernelImpl struct {
	kernel    func(left, right *float64, count int) float64
	name      string
	available bool
}

func pickGeometricProductKernel(
	candidates []geometricProductKernelImpl,
) func(left, right, destination *float64) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no geometric product kernel available")
}

func pickRotorSimilarityKernel(
	candidates []rotorSimilarityKernelImpl,
) func(left, right *float64, count int) float64 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no rotor similarity kernel available")
}
