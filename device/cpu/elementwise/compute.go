package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func loadF64(pointer unsafe.Pointer, index int) float64 {
	return *(*float64)(unsafe.Add(pointer, uintptr(index)*8))
}

func storeF64(pointer unsafe.Pointer, index int, value float64) {
	*(*float64)(unsafe.Add(pointer, uintptr(index)*8)) = value
}

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

func applyFloat64Binary(
	leftValue, rightValue float64,
	apply func(leftValue, rightValue float32) float32,
) float64 {
	return float64(apply(float32(leftValue), float32(rightValue)))
}

func dispatchBinary(
	dst, left, right unsafe.Pointer,
	count int,
	format dtype.DType,
	f32 func(dst, left, right unsafe.Pointer, count int),
	f16 func(dst, left, right unsafe.Pointer, count int),
	bf16 func(dst, left, right unsafe.Pointer, count int),
	apply func(leftValue, rightValue float32) float32,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		f32(dst, left, right, count)
	case dtype.Float64:
		for index := 0; index < count; index++ {
			leftValue := loadF64(left, index)
			rightValue := loadF64(right, index)
			storeF64(dst, index, applyFloat64Binary(leftValue, rightValue, apply))
		}
	case dtype.Float16:
		f16(dst, left, right, count)
	case dtype.BFloat16:
		bf16(dst, left, right, count)
	}
}

func dispatchUnary(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	f32 func(dst, src unsafe.Pointer, count int),
	f16 func(dst, src unsafe.Pointer, count int),
	bf16 func(dst, src unsafe.Pointer, count int),
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		f32(dst, src, count)
	case dtype.Float16:
		f16(dst, src, count)
	case dtype.BFloat16:
		bf16(dst, src, count)
	}
}

func dispatchAxpy(
	y, x unsafe.Pointer,
	count int,
	alpha float32,
	format dtype.DType,
	f32 func(y, x unsafe.Pointer, count int, alpha float32),
	f16 func(y, x unsafe.Pointer, count int, alpha float32),
	bf16 func(y, x unsafe.Pointer, count int, alpha float32),
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		f32(y, x, count, alpha)
	case dtype.Float16:
		f16(y, x, count, alpha)
	case dtype.BFloat16:
		bf16(y, x, count, alpha)
	}
}
