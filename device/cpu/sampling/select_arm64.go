//go:build arm64

package sampling

var greedySampleF32Funcs = []f32GreedyKernelImpl{
	{GreedySampleF32NEON, "neon", true},
	{greedySampleF32Generic, "generic", true},
}

var samplingSoftmaxRowF32Funcs = []f32SoftmaxRowKernelImpl{
	{SamplingSoftmaxRowF32NEON, "neon", true},
	{samplingSoftmaxRowF32Generic, "generic", true},
}
