//go:build amd64

package geometry

import "golang.org/x/sys/cpu"

var sumFloat64Funcs = []f64ReduceKernelImpl{
	{sumFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{sumFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{sumFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{sumFloat64Scalar, "generic", true},
}

var sumOfSquaresFloat64Funcs = []f64ReduceKernelImpl{
	{sumOfSquaresFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{sumOfSquaresFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{sumOfSquaresFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{sumOfSquaresFloat64Scalar, "generic", true},
}

var dotFloat64Funcs = []f64DotKernelImpl{
	{dotFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{dotFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{dotFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{dotFloat64Scalar, "generic", true},
}

var scaleFloat64Funcs = []f64ScaleKernelImpl{
	{scaleFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{scaleFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{scaleFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{scaleFloat64Scalar, "generic", true},
}

var addScalarFloat64Funcs = []f64AddScalarKernelImpl{
	{addScalarFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{addScalarFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{addScalarFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{addScalarFloat64Scalar, "generic", true},
}

var mulFloat64Funcs = []f64BinaryKernelImpl{
	{mulFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{mulFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{mulFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{mulFloat64Scalar, "generic", true},
}

var addFloat64Funcs = []f64BinaryKernelImpl{
	{addFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{addFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{addFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{addFloat64Scalar, "generic", true},
}

var sqrtFloat64Funcs = []f64UnaryKernelImpl{
	{sqrtFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{sqrtFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{sqrtFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{sqrtFloat64Scalar, "generic", true},
}

var maxFloat64Funcs = []f64ReduceKernelImpl{
	{maxFloat64AVX512, "avx512", cpu.X86.HasAVX512F},
	{maxFloat64AVX2, "avx2", cpu.X86.HasAVX2},
	{maxFloat64SSE2, "sse2", cpu.X86.HasSSE2},
	{maxFloat64Scalar, "generic", true},
}
