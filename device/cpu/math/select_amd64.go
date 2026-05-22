//go:build amd64

package math

import "golang.org/x/sys/cpu"

var invSqrtDimScaleF32Funcs = []f32InvSqrtDimScaleKernelImpl{
	{InvSqrtDimScaleF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{InvSqrtDimScaleF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{InvSqrtDimScaleF32SSE2, "sse2", cpu.X86.HasSSE2},
	{InvSqrtDimScaleGeneric, "generic", true},
}

var logSumExpF32Funcs = []f32LogSumExpKernelImpl{
	{LogSumExpF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{LogSumExpF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{LogSumExpF32SSE2, "sse2", cpu.X86.HasSSE2},
	{LogSumExpGeneric, "generic", true},
}

var outerF32Funcs = []f32OuterKernelImpl{
	{OuterF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{OuterF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{OuterF32SSE2, "sse2", cpu.X86.HasSSE2},
	{OuterGeneric, "generic", true},
}
