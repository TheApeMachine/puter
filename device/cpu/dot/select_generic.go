//go:build !amd64 && !arm64

package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

// Host-side convenience helpers — see select_amd64.go for design rationale.

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
		{DotF32Generic, "generic", true},
	}
	dotBF16Funcs = []bf16DotKernelImpl{
		{DotBF16Generic, "generic", true},
	}
	dotFP16Funcs = []fp16DotKernelImpl{
		{DotFP16Generic, "generic", true},
	}
	dotInt8Funcs = []int8DotKernelImpl{
		{DotInt8Generic, "generic", true},
	}
)
