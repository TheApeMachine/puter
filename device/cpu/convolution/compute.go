package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func float32View(pointer unsafe.Pointer, length int) []float32 {
	if length == 0 {
		return nil
	}

	return unsafe.Slice((*float32)(pointer), length)
}

func loadF32(pointer unsafe.Pointer, index int) float32 {
	return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
}

func storeF32(pointer unsafe.Pointer, index int, value float32) {
	*(*float32)(unsafe.Add(pointer, uintptr(index)*4)) = value
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

type elementLoad func(pointer unsafe.Pointer, index int) float32

type elementStore func(pointer unsafe.Pointer, index int, value float32)

func elementAccessors(format dtype.DType) (elementLoad, elementStore) {
	switch format {
	case dtype.Float16:
		return loadF16, storeF16
	case dtype.BFloat16:
		return loadBF16, storeBF16
	default:
		return loadF32, storeF32
	}
}
