//go:build !amd64 && !arm64

package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

// Host-side convenience helpers — see select_amd64.go for design rationale.

func SumFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.Sum(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
	return result
}

func SumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.Sum(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.BFloat16,
	)
	return dtype.NewBfloat16FromFloat32(result)
}

func SumFloat16Native(values []dtype.F16) dtype.F16 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.Sum(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float16,
	)
	return dtype.Fromfloat32(result)
}

func ReduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.Prod(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
	return result
}

func ReduceMinFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.ReduceMin(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
	return result
}

func ReduceMaxFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.ReduceMax(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
	return result
}

func L1NormFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	var result float32
	Default.L1Norm(
		unsafe.Pointer(&result),
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
	return result
}

var (
	sumF32Funcs = []f32ReduceKernelImpl{
		{SumF32Generic, "generic", true},
	}
	prodF32Funcs = []f32ReduceKernelImpl{
		{ProdF32Generic, "generic", true},
	}
	minF32Funcs = []f32ReduceKernelImpl{
		{MinF32Generic, "generic", true},
	}
	maxF32Funcs = []f32ReduceKernelImpl{
		{MaxF32Generic, "generic", true},
	}
	l1NormF32Funcs = []f32ReduceKernelImpl{
		{L1NormF32Generic, "generic", true},
	}
	sumBF16Funcs = []bf16SumKernelImpl{
		{SumBF16Generic, "generic", true},
	}
	sumFP16Funcs = []fp16SumKernelImpl{
		{SumFP16Generic, "generic", true},
	}
	prodBF16Funcs = []bf16ProdKernelImpl{
		{ProdBF16Generic, "generic", true},
	}
	prodFP16Funcs = []fp16ProdKernelImpl{
		{ProdFP16Generic, "generic", true},
	}
	minBF16Funcs = []bf16MinKernelImpl{
		{MinBF16Generic, "generic", true},
	}
	maxBF16Funcs = []bf16MaxKernelImpl{
		{MaxBF16Generic, "generic", true},
	}
	l1NormBF16Funcs = []bf16L1NormKernelImpl{
		{L1NormBF16Generic, "generic", true},
	}
	minFP16Funcs = []fp16MinKernelImpl{
		{MinFP16Generic, "generic", true},
	}
	maxFP16Funcs = []fp16MaxKernelImpl{
		{MaxFP16Generic, "generic", true},
	}
	l1NormFP16Funcs = []fp16L1NormKernelImpl{
		{L1NormFP16Generic, "generic", true},
	}
)
