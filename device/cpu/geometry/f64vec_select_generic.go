//go:build !amd64 && !arm64

package geometry

var sumFloat64Funcs = []f64ReduceKernelImpl{
	{sumFloat64Scalar, "generic", true},
}

var sumOfSquaresFloat64Funcs = []f64ReduceKernelImpl{
	{sumOfSquaresFloat64Scalar, "generic", true},
}

var dotFloat64Funcs = []f64DotKernelImpl{
	{dotFloat64Scalar, "generic", true},
}

var scaleFloat64Funcs = []f64ScaleKernelImpl{
	{scaleFloat64Scalar, "generic", true},
}

var addScalarFloat64Funcs = []f64AddScalarKernelImpl{
	{addScalarFloat64Scalar, "generic", true},
}

var mulFloat64Funcs = []f64BinaryKernelImpl{
	{mulFloat64Scalar, "generic", true},
}

var addFloat64Funcs = []f64BinaryKernelImpl{
	{addFloat64Scalar, "generic", true},
}

var sqrtFloat64Funcs = []f64UnaryKernelImpl{
	{sqrtFloat64Scalar, "generic", true},
}

var maxFloat64Funcs = []f64ReduceKernelImpl{
	{maxFloat64Scalar, "generic", true},
}
