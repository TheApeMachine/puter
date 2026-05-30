//go:build arm64

package geometry

var phaseCouplingFloat32Funcs = []phaseCouplingKernelImpl{
	{PhaseCouplingFloat32NEON, "neon", true},
	{PhaseCouplingFloat32ScalarDispatch, "generic", true},
}

var phaseCouplingFloat16Funcs = []phaseCouplingUInt16KernelImpl{
	{PhaseCouplingFloat16NEON, "neon", true},
	{PhaseCouplingFloat16ScalarDispatch, "generic", true},
}

var phaseCouplingBFloat16Funcs = []phaseCouplingUInt16KernelImpl{
	{PhaseCouplingBFloat16NEON, "neon", true},
	{PhaseCouplingBFloat16ScalarDispatch, "generic", true},
}

func PhaseCouplingFloat32ScalarDispatch(
	destination, leftGrowth, rightGrowth []float32,
	count int,
) {
	PhaseCouplingFloat32Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}

func PhaseCouplingFloat16ScalarDispatch(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	PhaseCouplingFloat16Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}

func PhaseCouplingBFloat16ScalarDispatch(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	PhaseCouplingBFloat16Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}
