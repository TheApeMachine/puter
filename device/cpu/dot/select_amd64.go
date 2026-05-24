//go:build amd64

package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"golang.org/x/sys/cpu"
)

// Host-side convenience helpers. These bridge the zero-host-sync
// device interface (`Default.Dot(dst, left, right, ...)`) to a
// Go-scalar-returning shape for host callers (tests, non-async setup).
// Device dispatch in the execution loop calls Default.Dot directly
// with a workspace destination.

func DotFloat32Native(left, right []float32) float32 {
	if len(left) == 0 {
		return 0
	}

	var result float32
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Float32,
	)
	return result
}

func DotBFloat16Native(left, right []dtype.BF16) dtype.BF16 {
	if len(left) == 0 {
		return 0
	}

	var result float32
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.BFloat16,
	)
	return dtype.NewBfloat16FromFloat32(result)
}

func DotFloat16Native(left, right []dtype.F16) dtype.F16 {
	if len(left) == 0 {
		return 0
	}

	var result float32
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Float16,
	)
	return dtype.Fromfloat32(result)
}

func DotInt8Native(left, right []int8) int32 {
	if len(left) == 0 {
		return 0
	}

	var result float32
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Int8,
	)
	return int32(result)
}

var (
	dotF32Funcs = []f32DotKernelImpl{
		{DotF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{DotF32Generic, "generic", true},
	}
	dotBF16Funcs = []bf16DotKernelImpl{
		{DotBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{DotBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{DotBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{DotBF16Generic, "generic", true},
	}
	dotFP16Funcs = []fp16DotKernelImpl{
		{DotFP16AVX512, "avx512", cpu.X86.HasAVX512F},
		{DotFP16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{DotFP16SSE2, "sse2", cpu.X86.HasSSE2},
		{DotFP16Generic, "generic", true},
	}
	dotInt8Funcs = []int8DotKernelImpl{
		{DotInt8AVX512, "avx512", cpu.X86.HasAVX512F},
		{DotInt8AVX2, "avx2", cpu.X86.HasAVX2},
		{DotInt8SSE2, "sse2", cpu.X86.HasSSE2},
		{DotInt8Generic, "generic", true},
	}
)
