package elementwise

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func uint16Views(dst, src *uint16, count int) (destination, source []uint16) {
	return unsafe.Slice(dst, count), unsafe.Slice(src, count)
}

func uint16BinaryViews(dst, left, right *uint16, count int) (destination, leftView, rightView []uint16) {
	return unsafe.Slice(dst, count), unsafe.Slice(left, count), unsafe.Slice(right, count)
}

func AddF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()
		destination[index] = dtype.Fromfloat32(leftValue + rightValue).Bits()
	}
}

func SubF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()
		destination[index] = dtype.Fromfloat32(leftValue - rightValue).Bits()
	}
}

func MulF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()
		destination[index] = dtype.Fromfloat32(leftValue * rightValue).Bits()
	}
}

func DivF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()
		destination[index] = dtype.Fromfloat32(leftValue / rightValue).Bits()
	}
}

func MaxF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()

		if leftValue > rightValue {
			destination[index] = leftView[index]
			continue
		}

		destination[index] = rightView[index]
	}
}

func MinF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := dtype.Frombits(leftView[index]).Float32()
		rightValue := dtype.Frombits(rightView[index]).Float32()

		if leftValue < rightValue {
			destination[index] = leftView[index]
			continue
		}

		destination[index] = rightView[index]
	}
}

func AbsF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := dtype.Frombits(source[index]).Float32()
		destination[index] = dtype.Fromfloat32(float32(math.Abs(float64(value)))).Bits()
	}
}

func NegF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := dtype.Frombits(source[index]).Float32()
		destination[index] = dtype.Fromfloat32(-value).Bits()
	}
}

func SqrtF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := dtype.Frombits(source[index]).Float32()
		destination[index] = dtype.Fromfloat32(float32(math.Sqrt(float64(value)))).Bits()
	}
}

func ReluF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := dtype.Frombits(source[index]).Float32()

		if value > 0 {
			destination[index] = source[index]
			continue
		}

		destination[index] = dtype.Fromfloat32(0).Bits()
	}
}

func AxpyF16Generic(y, x *uint16, alpha float32, count int) {
	destination := unsafe.Slice(y, count)
	source := unsafe.Slice(x, count)

	for index := 0; index < count; index++ {
		yValue := dtype.Frombits(destination[index]).Float32()
		xValue := dtype.Frombits(source[index]).Float32()
		destination[index] = dtype.Fromfloat32(yValue + alpha*xValue).Bits()
	}
}

func bf16ToFloat32(bits uint16) float32 {
	value := dtype.BF16(bits)
	return (&value).Float32()
}

func AddBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(leftValue + rightValue))
	}
}

func SubBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(leftValue - rightValue))
	}
}

func MulBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(leftValue * rightValue))
	}
}

func DivBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(leftValue / rightValue))
	}
}

func MaxBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])

		if leftValue > rightValue {
			destination[index] = leftView[index]
			continue
		}

		destination[index] = rightView[index]
	}
}

func MinBF16Generic(dst, left, right *uint16, count int) {
	destination, leftView, rightView := uint16BinaryViews(dst, left, right, count)

	for index := 0; index < count; index++ {
		leftValue := bf16ToFloat32(leftView[index])
		rightValue := bf16ToFloat32(rightView[index])

		if leftValue < rightValue {
			destination[index] = leftView[index]
			continue
		}

		destination[index] = rightView[index]
	}
}

func AbsBF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := bf16ToFloat32(source[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(float32(math.Abs(float64(value)))))
	}
}

func NegBF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := bf16ToFloat32(source[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(-value))
	}
}

func SqrtBF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := bf16ToFloat32(source[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(float32(math.Sqrt(float64(value)))))
	}
}

func ReluBF16Generic(dst, src *uint16, count int) {
	destination, source := uint16Views(dst, src, count)

	for index := 0; index < count; index++ {
		value := bf16ToFloat32(source[index])

		if value > 0 {
			destination[index] = source[index]
			continue
		}

		destination[index] = uint16(dtype.NewBfloat16FromFloat32(0))
	}
}

func AxpyBF16Generic(y, x *uint16, alpha float32, count int) {
	destination := unsafe.Slice(y, count)
	source := unsafe.Slice(x, count)

	for index := 0; index < count; index++ {
		yValue := bf16ToFloat32(destination[index])
		xValue := bf16ToFloat32(source[index])
		destination[index] = uint16(dtype.NewBfloat16FromFloat32(yValue + alpha*xValue))
	}
}
