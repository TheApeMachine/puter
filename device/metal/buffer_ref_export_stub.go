//go:build !darwin || !cgo

package metal

import "unsafe"

/*
BufferRefFromDispatch is unavailable off Darwin.
*/
func BufferRefFromDispatch(pointer unsafe.Pointer) uintptr {
	_ = pointer

	return 0
}
