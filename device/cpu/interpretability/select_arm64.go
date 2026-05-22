//go:build arm64

package interpretability

var activationSteerFloat32Funcs = []activationSteerKernelImpl{
	{activationSteerFloat32NEON, "neon", true},
	{activationSteerFloat32Scalar, "generic", true},
}

func activationSteerFloat32NEON(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32NEON(
		&destination[0], &base[0], &direction[0], coefficient, count,
	)
}

func activationSteerFloat32Scalar(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32Scalar(destination[:count], base[:count], direction[:count], coefficient)
}
