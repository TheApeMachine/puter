package attention

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

func loadTyped(pointer unsafe.Pointer, index int, format dtype.DType) float32 {
	switch format {
	case dtype.Float32:
		return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
	case dtype.Float16:
		return loadF16(pointer, index)
	case dtype.BFloat16:
		return loadBF16(pointer, index)
	default:
		panic("attention: unsupported dtype for load")
	}
}

func storeTyped(pointer unsafe.Pointer, index int, value float32, format dtype.DType) {
	switch format {
	case dtype.Float32:
		*(*float32)(unsafe.Add(pointer, uintptr(index)*4)) = value
	case dtype.Float16:
		storeF16(pointer, index, value)
	case dtype.BFloat16:
		storeBF16(pointer, index, value)
	default:
		panic("attention: unsupported dtype for store")
	}
}

func typedElementPointer(base unsafe.Pointer, index int, format dtype.DType) unsafe.Pointer {
	switch format {
	case dtype.Float32:
		return unsafe.Add(base, uintptr(index)*4)
	case dtype.Float16, dtype.BFloat16:
		return unsafe.Add(base, uintptr(index)*2)
	default:
		panic("attention: unsupported dtype for pointer offset")
	}
}
