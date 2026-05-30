//go:build arm64

package geometry

var geometricProductFuncs = []geometricProductKernelImpl{
	{geometricProductFloat64NEON, "neon", true},
	{geometricProductFloat64Scalar, "generic", true},
}

var rotorSimilarityFuncs = []rotorSimilarityKernelImpl{
	{rotorSimilarity128NEON, "neon", true},
	{rotorSimilarity128Scalar, "generic", true},
}
