package scalar

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func StoreFloat32(dst unsafe.Pointer, value float32, format dtype.DType) {
	switch format {
	case dtype.Float64:
		*(*float64)(dst) = float64(value)
	case dtype.Float32:
		*(*float32)(dst) = value
	case dtype.Float16:
		*(*uint16)(dst) = uint16(dtype.Fromfloat32(value))
	case dtype.BFloat16:
		*(*uint16)(dst) = uint16(dtype.NewBfloat16FromFloat32(value))
	default:
		panic("scalar: unsupported floating scalar dtype")
	}
}

func StoreInt32(dst unsafe.Pointer, value int32) {
	*(*int32)(dst) = value
}
