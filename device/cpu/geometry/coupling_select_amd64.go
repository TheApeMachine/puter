//go:build amd64

package geometry

import "golang.org/x/sys/cpu"

var phaseCouplingFloat32Funcs = []phaseCouplingKernelImpl{
	{PhaseCouplingFloat32AVX512, "avx512", cpu.X86.HasAVX512F},
	{PhaseCouplingFloat32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{PhaseCouplingFloat32SSE2, "sse2", cpu.X86.HasSSE2},
	{PhaseCouplingFloat32ScalarDispatch, "generic", true},
}

var phaseCouplingFloat16Funcs = []phaseCouplingUInt16KernelImpl{
	{PhaseCouplingFloat16AVX512, "avx512fp16", hasAVX512FP16},
	{PhaseCouplingFloat16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{PhaseCouplingFloat16SSE2, "sse2", cpu.X86.HasSSE2},
	{PhaseCouplingFloat16ScalarDispatch, "generic", true},
}

var phaseCouplingBFloat16Funcs = []phaseCouplingUInt16KernelImpl{
	{PhaseCouplingBFloat16AVX512, "avx512", cpu.X86.HasAVX512F},
	{PhaseCouplingBFloat16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
	{PhaseCouplingBFloat16SSE2, "sse2", cpu.X86.HasSSE2},
	{PhaseCouplingBFloat16ScalarDispatch, "generic", true},
}

func PhaseCouplingFloat32ScalarDispatch(
	destination, leftGrowth, rightGrowth []float32,
	count int,
) {
	PhaseCouplingFloat32Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}

func PhaseCouplingFloat16ScalarDispatch(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	PhaseCouplingFloat16Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}

func PhaseCouplingBFloat16ScalarDispatch(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	PhaseCouplingBFloat16Scalar(destination[:count], leftGrowth[:count], rightGrowth[:count])
}
