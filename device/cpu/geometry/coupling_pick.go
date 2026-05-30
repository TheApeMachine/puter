package geometry

type phaseCouplingKernelImpl struct {
	kernel    func(destination, leftGrowth, rightGrowth []float32, count int)
	name      string
	available bool
}

type phaseCouplingUInt16KernelImpl struct {
	kernel    func(destination, leftGrowth, rightGrowth []uint16, count int)
	name      string
	available bool
}

func pickPhaseCouplingFloat32Kernel(
	candidates []phaseCouplingKernelImpl,
) func(destination, leftGrowth, rightGrowth []float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no float32 phase coupling kernel available")
}

func pickPhaseCouplingUInt16Kernel(
	candidates []phaseCouplingUInt16KernelImpl,
) func(destination, leftGrowth, rightGrowth []uint16, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("geometry: no uint16 phase coupling kernel available")
}
