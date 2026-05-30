//go:build amd64

package geometry

import "golang.org/x/sys/cpu"

var geometricProductFuncs = []geometricProductKernelImpl{
	{geometricProductFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{geometricProductFloat64Scalar, "generic", true},
}

var rotorSimilarityFuncs = []rotorSimilarityKernelImpl{
	{rotorSimilarity128SSE2, "sse2", cpu.X86.HasSSE2},
	{rotorSimilarity128Scalar, "generic", true},
}
