//go:build arm64

package math

var invSqrtDimScaleF32Funcs = []f32InvSqrtDimScaleKernelImpl{
	{InvSqrtDimScaleF32NEON, "neon", true},
	{InvSqrtDimScaleGeneric, "generic", true},
}

var logSumExpF32Funcs = []f32LogSumExpKernelImpl{
	{LogSumExpF32NEON, "neon", true},
	{LogSumExpGeneric, "generic", true},
}

var outerF32Funcs = []f32OuterKernelImpl{
	{OuterF32NEON, "neon", true},
	{OuterGeneric, "generic", true},
}
