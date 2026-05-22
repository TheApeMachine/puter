//go:build arm64

package model_editing

var weightGraftAddFloat32Funcs = []weightGraftAddKernelImpl{
	{weightGraftAddFloat32NEON, "neon", true},
	{weightGraftAddFloat32Scalar, "generic", true},
}

func weightGraftAddFloat32NEON(weights, injection []float32, count int) {
	WeightGraftAddFloat32NEON(&weights[0], &injection[0], count)
}

func weightGraftAddFloat32Scalar(weights, injection []float32, count int) {
	WeightGraftAddFloat32Scalar(weights[:count], injection[:count])
}
