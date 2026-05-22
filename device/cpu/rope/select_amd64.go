//go:build amd64

package rope

import "golang.org/x/sys/cpu"

var ropePairsF32Funcs = []f32RopePairsKernelImpl{
	{ropePairsF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{ropePairsF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{ropePairsF32SSE2, "sse2", cpu.X86.HasSSE2},
	{ropePairsF32Generic, "generic", true},
}

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	ropePairsF32Kernel(out, in, cosBuf, sinBuf)
}

func ropePairsF32Generic(out, in, cosBuf, sinBuf []float32) {
	RopePairsGeneric(out, in, cosBuf, sinBuf)
}
