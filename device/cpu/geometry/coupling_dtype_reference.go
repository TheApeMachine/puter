package geometry

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

const phaseCouplingEpsF16 = dtype.F16(0x211f)

const phaseCouplingEpsBF16 = dtype.BF16(0x3c23)

func f16Abs(value dtype.F16) dtype.F16 {
	return dtype.Frombits(value.Bits() & 0x7fff)
}

func f16Mul(left, right dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(left.Float32() * right.Float32())
}

func f16Div(numerator, denominator dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(numerator.Float32() / denominator.Float32())
}

func f16Sqrt(value dtype.F16) dtype.F16 {
	return dtype.Fromfloat32(float32(math.Sqrt(float64(value.Float32()))))
}

func f16Lt(left, right dtype.F16) bool {
	return left.Float32() < right.Float32()
}

func scalarPhaseCouplingReferenceF16(
	leftValue, rightValue dtype.F16,
) dtype.F16 {
	absLeft := f16Abs(leftValue)
	absRight := f16Abs(rightValue)
	geometricMean := f16Sqrt(f16Mul(absLeft, absRight))

	if f16Lt(geometricMean, phaseCouplingEpsF16) {
		return dtype.F16(0)
	}

	numerator := f16Mul(leftValue, rightValue)
	denominator := f16Mul(geometricMean, geometricMean)

	return f16Div(numerator, denominator)
}

func bf16Abs(value dtype.BF16) dtype.BF16 {
	return dtype.BF16(value.Bits() & 0x7fff)
}

func bf16Mul(left, right dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(left.Float32() * right.Float32())
}

func bf16Div(numerator, denominator dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(numerator.Float32() / denominator.Float32())
}

func bf16Sqrt(value dtype.BF16) dtype.BF16 {
	return dtype.NewBfloat16FromFloat32(float32(math.Sqrt(float64(value.Float32()))))
}

func bf16Lt(left, right dtype.BF16) bool {
	return left.Float32() < right.Float32()
}

func scalarPhaseCouplingReferenceBF16(
	leftValue, rightValue dtype.BF16,
) dtype.BF16 {
	absLeft := bf16Abs(leftValue)
	absRight := bf16Abs(rightValue)
	geometricMean := bf16Sqrt(bf16Mul(absLeft, absRight))

	if bf16Lt(geometricMean, phaseCouplingEpsBF16) {
		return dtype.BF16(0)
	}

	numerator := bf16Mul(leftValue, rightValue)
	denominator := bf16Mul(geometricMean, geometricMean)

	return bf16Div(numerator, denominator)
}
