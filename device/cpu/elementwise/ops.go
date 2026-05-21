package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	if format == dtype.Float64 {
		if count == 0 {
			return
		}

		runAddF64(dst, left, right, count)

		return
	}

	dispatchBinary(
		dst, left, right, count, format, runAddF32, runAddF16, runAddBF16,
		func(leftValue, rightValue float32) float32 { return leftValue + rightValue },
	)
}

func Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runSubF32, runSubF16, runSubBF16,
		func(leftValue, rightValue float32) float32 { return leftValue - rightValue },
	)
}

func Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMulF32, runMulF16, runMulBF16,
		func(leftValue, rightValue float32) float32 { return leftValue * rightValue },
	)
}

func Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runDivF32, runDivF16, runDivBF16,
		func(leftValue, rightValue float32) float32 { return leftValue / rightValue },
	)
}

func Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMaxF32, runMaxF16, runMaxBF16,
		func(leftValue, rightValue float32) float32 {
			if leftValue > rightValue {
				return leftValue
			}

			return rightValue
		},
	)
}

func Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMinF32, runMinF16, runMinBF16,
		func(leftValue, rightValue float32) float32 {
			if leftValue < rightValue {
				return leftValue
			}

			return rightValue
		},
	)
}

func Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(dst, src, count, format, runAbsF32, runAbsF16, runAbsBF16)
}

func Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(dst, src, count, format, runNegF32, runNegF16, runNegBF16)
}

func Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(dst, src, count, format, runSqrtF32, runSqrtF16, runSqrtBF16)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(dst, src, count, format, runReluF32, runReluF16, runReluBF16)
}

func Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	dispatchAxpy(y, x, count, alpha, format, runAxpyF32, runAxpyF16, runAxpyBF16)
}
