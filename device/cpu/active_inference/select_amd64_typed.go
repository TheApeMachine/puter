//go:build amd64

package active_inference

import "golang.org/x/sys/cpu"

var (
	freeEnergyBF16Funcs = []bf16FreeEnergyKernelImpl{
		{FreeEnergyBFloat16Generic, "generic", true},
	}
	expectedFreeEnergyBF16Funcs = []bf16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyBFloat16Generic, "generic", true},
	}
	beliefUpdateBF16Funcs = []bf16BeliefUpdateKernelImpl{
		{BeliefUpdateBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{BeliefUpdateBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{BeliefUpdateBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{BeliefUpdateBFloat16Generic, "generic", true},
	}
	precisionWeightBF16Funcs = []bf16PrecisionWeightKernelImpl{
		{PrecisionWeightBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{PrecisionWeightBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{PrecisionWeightBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{PrecisionWeightBFloat16Generic, "generic", true},
	}

	freeEnergyFP16Funcs = []fp16FreeEnergyKernelImpl{
		{FreeEnergyFloat16Generic, "generic", true},
	}
	expectedFreeEnergyFP16Funcs = []fp16ExpectedFreeEnergyKernelImpl{
		{ExpectedFreeEnergyFloat16Generic, "generic", true},
	}
	beliefUpdateFP16Funcs = []fp16BeliefUpdateKernelImpl{
		{BeliefUpdateFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{BeliefUpdateFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{BeliefUpdateFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{BeliefUpdateFloat16Generic, "generic", true},
	}
	precisionWeightFP16Funcs = []fp16PrecisionWeightKernelImpl{
		{PrecisionWeightFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{PrecisionWeightFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{PrecisionWeightFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{PrecisionWeightFloat16Generic, "generic", true},
	}
)
