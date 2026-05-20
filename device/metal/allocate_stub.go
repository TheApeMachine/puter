//go:build !darwin || !cgo

package metal

import "unsafe"

func metalBufferContents(buffer uintptr) unsafe.Pointer {
	_ = buffer

	return nil
}

func metalMemset(destination unsafe.Pointer, value byte, byteCount int) {
	_ = destination
	_ = value
	_ = byteCount
}
