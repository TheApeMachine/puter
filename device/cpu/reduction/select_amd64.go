//go:build amd64

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"golang.org/x/sys/cpu"
)

func SumFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func SumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.NewBfloat16FromFloat32(Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.BFloat16,
	))
}

func SumFloat16Native(values []dtype.F16) dtype.F16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.Fromfloat32(Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float16,
	))
}

func ReduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return Prod(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func ReduceMinFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return ReduceMin(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func ReduceMaxFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return ReduceMax(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func L1NormFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return L1Norm(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

var (
	sumF32Funcs = []f32ReduceKernelImpl{
		{SumF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SumF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SumF32SSE2, "sse2", cpu.X86.HasSSE2},
		{SumF32Generic, "generic", true},
	}
	prodF32Funcs = []f32ReduceKernelImpl{
		{ProdF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ProdF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ProdF32SSE2, "sse2", cpu.X86.HasSSE2},
		{ProdF32Generic, "generic", true},
	}
	minF32Funcs = []f32ReduceKernelImpl{
		{MinF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MinF32SSE2, "sse2", cpu.X86.HasSSE2},
		{MinF32Generic, "generic", true},
	}
	maxF32Funcs = []f32ReduceKernelImpl{
		{MaxF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MaxF32SSE2, "sse2", cpu.X86.HasSSE2},
		{MaxF32Generic, "generic", true},
	}
	l1NormF32Funcs = []f32ReduceKernelImpl{
		{L1NormF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{L1NormF32AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{L1NormF32SSE2, "sse2", cpu.X86.HasSSE2},
		{L1NormF32Generic, "generic", true},
	}
	sumBF16Funcs = []bf16SumKernelImpl{
		{SumBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SumBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SumBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{SumBF16Generic, "generic", true},
	}
	sumFP16Funcs = []fp16SumKernelImpl{
		{SumFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SumFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SumFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{SumFP16Generic, "generic", true},
	}
	prodBF16Funcs = []bf16ProdKernelImpl{
		{ProdBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{ProdBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ProdBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{ProdBF16Generic, "generic", true},
	}
	prodFP16Funcs = []fp16ProdKernelImpl{
		{ProdFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{ProdFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ProdFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{ProdFP16Generic, "generic", true},
	}
	minBF16Funcs = []bf16MinKernelImpl{
		{MinBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MinBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MinBF16Generic, "generic", true},
	}
	maxBF16Funcs = []bf16MaxKernelImpl{
		{MaxBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MaxBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MaxBF16Generic, "generic", true},
	}
	l1NormBF16Funcs = []bf16L1NormKernelImpl{
		{L1NormBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{L1NormBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{L1NormBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{L1NormBF16Generic, "generic", true},
	}
	minFP16Funcs = []fp16MinKernelImpl{
		{MinFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MinFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{MinFP16Generic, "generic", true},
	}
	maxFP16Funcs = []fp16MaxKernelImpl{
		{MaxFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MaxFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{MaxFP16Generic, "generic", true},
	}
	l1NormFP16Funcs = []fp16L1NormKernelImpl{
		{L1NormFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{L1NormFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{L1NormFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{L1NormFP16Generic, "generic", true},
	}
)
