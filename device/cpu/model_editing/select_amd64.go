//go:build amd64

package model_editing

import "golang.org/x/sys/cpu"

var weightGraftAddFloat32Funcs = []weightGraftAddKernelImpl{
	{weightGraftAddFloat32AVX512, "avx512", cpu.X86.HasAVX512F},
	{weightGraftAddFloat32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{weightGraftAddFloat32SSE2, "sse2", cpu.X86.HasSSE2},
	{weightGraftAddFloat32Scalar, "generic", true},
}

func weightGraftAddFloat32AVX512(weights, injection []float32, count int) {
	WeightGraftAddFloat32AVX512(&weights[0], &injection[0], count)
}

func weightGraftAddFloat32Scalar(weights, injection []float32, count int) {
	WeightGraftAddFloat32Scalar(weights[:count], injection[:count])
}
