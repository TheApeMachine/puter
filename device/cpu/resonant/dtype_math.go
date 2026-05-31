package resonant

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

var (
	resonantNormalizeEpsilonF16  = dtype.Fromfloat32(1e-6)
	resonantNormalizeEpsilonBF16 = dtype.NewBfloat16FromFloat32(1e-6)
)

func resonantInvDimF16(headDim int) dtype.F16 {
	return dtype.Fromfloat32(float32(1.0) / float32(headDim))
}

func resonantInvDimBF16(headDim int) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(float32(1.0) / float32(headDim))
}

func resonantScaleF16(scale float32) dtype.F16 {
	return dtype.Fromfloat32(scale)
}

func resonantScaleBF16(scale float32) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(scale)
}

func resonantDampingF16(damping float32) dtype.F16 {
	return dtype.Fromfloat32(damping)
}

func resonantDampingBF16(damping float32) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(damping)
}

func resonantOneMinusF16(damping float32) dtype.F16 {
	return dtype.Fromfloat32(float32(1.0) - damping)
}

func resonantOneMinusBF16(damping float32) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(float32(1.0) - damping)
}

func resonantAddF16(left, right dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(left.Float32() + right.Float32())
}

func resonantSubF16(left, right dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(left.Float32() - right.Float32())
}

func resonantMulF16(left, right dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(left.Float32() * right.Float32())
}

func resonantDivF16(numerator, denominator dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(numerator.Float32() / denominator.Float32())
}

func resonantSqrtF16(value dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(float32(math.Sqrt(float64(value.Float32()))))
}

func resonantInvRadiusF16(accumReal, accumImag dtype.F16) dtype.F16 {
	sumSquares := resonantAddF16(
		resonantMulF16(accumReal, accumReal),
		resonantMulF16(accumImag, accumImag),
	)
	sumSquares = resonantAddF16(sumSquares, resonantNormalizeEpsilonF16)

	return resonantDivF16(dtype.Fromfloat32(1), resonantSqrtF16(sumSquares))
}

func resonantAddBF16(left, right dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(left.Float32() + right.Float32())
}

func resonantSubBF16(left, right dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(left.Float32() - right.Float32())
}

func resonantMulBF16(left, right dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(left.Float32() * right.Float32())
}

func resonantDivBF16(numerator, denominator dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(numerator.Float32() / denominator.Float32())
}

func resonantSqrtBF16(value dtype.BF16) dtype.BF16 {
	if value.Float32() <= 0 {
		return dtype.BF16(0)
	}

	guess := value

	for range 5 {
		half := dtype.NewBfloat16FromFloat32(0.5)
		guess = resonantMulBF16(
			half,
			resonantAddBF16(guess, resonantDivBF16(value, guess)),
		)
	}

	return guess
}

func resonantInvRadiusBF16(accumReal, accumImag dtype.BF16) dtype.BF16 {
	sumSquares := resonantAddBF16(
		resonantMulBF16(accumReal, accumReal),
		resonantMulBF16(accumImag, accumImag),
	)
	sumSquares = resonantAddBF16(sumSquares, resonantNormalizeEpsilonBF16)

	return resonantDivBF16(dtype.NewBfloat16FromFloat32(1), resonantSqrtBF16(sumSquares))
}
