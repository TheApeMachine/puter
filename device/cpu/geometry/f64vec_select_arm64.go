//go:build arm64

package geometry

var sumFloat64Funcs = []f64ReduceKernelImpl{
	{sumFloat64NEON, "neon", true},
	{sumFloat64Scalar, "generic", true},
}

var sumOfSquaresFloat64Funcs = []f64ReduceKernelImpl{
	{sumOfSquaresFloat64NEON, "neon", true},
	{sumOfSquaresFloat64Scalar, "generic", true},
}

var dotFloat64Funcs = []f64DotKernelImpl{
	{dotFloat64NEON, "neon", true},
	{dotFloat64Scalar, "generic", true},
}

var scaleFloat64Funcs = []f64ScaleKernelImpl{
	{scaleFloat64NEON, "neon", true},
	{scaleFloat64Scalar, "generic", true},
}

var addScalarFloat64Funcs = []f64AddScalarKernelImpl{
	{addScalarFloat64NEON, "neon", true},
	{addScalarFloat64Scalar, "generic", true},
}

var mulFloat64Funcs = []f64BinaryKernelImpl{
	{mulFloat64NEON, "neon", true},
	{mulFloat64Scalar, "generic", true},
}

var addFloat64Funcs = []f64BinaryKernelImpl{
	{addFloat64NEON, "neon", true},
	{addFloat64Scalar, "generic", true},
}

var sqrtFloat64Funcs = []f64UnaryKernelImpl{
	{sqrtFloat64NEON, "neon", true},
	{sqrtFloat64Scalar, "generic", true},
}

var maxFloat64Funcs = []f64ReduceKernelImpl{
	{maxFloat64NEON, "neon", true},
	{maxFloat64Scalar, "generic", true},
}
