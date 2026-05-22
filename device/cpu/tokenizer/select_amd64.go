//go:build amd64

package tokenizer

import "golang.org/x/sys/cpu"

var packInt32Funcs = []int32PackKernelImpl{
	{TokenizerPackInt32AVX512, "avx512", cpu.X86.HasAVX512F},
	{TokenizerPackInt32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{TokenizerPackInt32SSE2, "sse2", cpu.X86.HasSSE2},
	{packInt32Generic, "generic", true},
}
