//go:build !amd64 && !arm64

package geometry

var geometricProductFuncs = []geometricProductKernelImpl{
	{geometricProductFloat64Scalar, "generic", true},
}

var rotorSimilarityFuncs = []rotorSimilarityKernelImpl{
	{rotorSimilarity128Scalar, "generic", true},
}
