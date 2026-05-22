//go:build amd64

package losses

import "golang.org/x/sys/cpu"

func MseSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return mseSumF32Kernel(&predictions[0], &targets[0], len(predictions))
}

func MaeSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return maeSumF32Kernel(&predictions[0], &targets[0], len(predictions))
}

var mseSumF32Funcs = []f32PairSumKernelImpl{
	{MseSumF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{MseSumF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{MseSumF32SSE2, "sse2", cpu.X86.HasSSE2},
	{MseSumF32Generic, "generic", true},
}

var maeSumF32Funcs = []f32PairSumKernelImpl{
	{MaeSumF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{MaeSumF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{MaeSumF32SSE2, "sse2", cpu.X86.HasSSE2},
	{MaeSumF32Generic, "generic", true},
}
