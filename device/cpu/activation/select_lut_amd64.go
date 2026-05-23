//go:build amd64

package activation

import "golang.org/x/sys/cpu"

var f16LUTGatherFuncs = []lutGatherImpl{
	{ApplyF16LUTAVX512, "avx512", cpu.X86.HasAVX512F},
	{ApplyF16LUTAVX2, "avx2", cpu.X86.HasAVX2},
	{ApplyF16LUTSSE2, "sse2", cpu.X86.HasSSE2},
	{applyF16LUTScalar, "generic", true},
}
