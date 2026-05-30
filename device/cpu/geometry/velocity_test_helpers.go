package geometry

import "unsafe"

func unsafePointerFromFloat32Slice(values []float32) unsafe.Pointer {
	return unsafe.Pointer(&values[0])
}

func unsafePointerFromUInt16Slice(values []uint16) unsafe.Pointer {
	return unsafe.Pointer(&values[0])
}
