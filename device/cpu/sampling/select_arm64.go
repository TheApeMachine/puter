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

func GreedySampleFloat32Native(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}

	return greedySampleF32Kernel(&logits[0], len(logits))
}

func SamplingSoftmaxRowFloat32Native(logits, out []float32, temperature float32) {
	if len(logits) == 0 {
		return
	}

	samplingSoftmaxRowF32Kernel(&logits[0], &out[0], temperature, len(logits))
}
