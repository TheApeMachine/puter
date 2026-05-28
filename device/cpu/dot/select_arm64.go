package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func DotF32NEON(left, right *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return DotFloat32NEONAsm(left, right, count)
}

func DotBF16NEON(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotBFloat16NEONAsm(left, right, count)
}

func DotFP16NEON(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotFloat16NEONAsm(left, right, count)
}

func DotInt8NEON(left, right *int8, count int) int32 {
	if count == 0 {
		return 0
	}

	return DotInt8NEONAsm(left, right, count)
}

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

	var result dtype.BF16
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.BFloat16,
	)
	return result
}

func DotFloat16Native(left, right []dtype.F16) dtype.F16 {
	if len(left) == 0 {
		return 0
	}

	var result dtype.F16
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Float16,
	)
	return result
}

func DotInt8Native(left, right []int8) int32 {
	if len(left) == 0 {
		return 0
	}

	var result int32
	Default.Dot(
		unsafe.Pointer(&result),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Int8,
	)
	return result
}

var (
	dotF32Funcs = []f32DotKernelImpl{
		{DotF32NEON, "neon", true},
		{DotF32Generic, "generic", true},
	}
	dotBF16Funcs = []bf16DotKernelImpl{
		{DotBF16NEON, "neon", true},
		{DotBF16Generic, "generic", true},
	}
	dotFP16Funcs = []fp16DotKernelImpl{
		{DotFP16NEON, "neon", true},
		{DotFP16Generic, "generic", true},
	}
	dotInt8Funcs = []int8DotKernelImpl{
		{DotInt8NEON, "neon", true},
		{DotInt8Generic, "generic", true},
	}
)
