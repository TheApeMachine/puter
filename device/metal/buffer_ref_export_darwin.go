//go:build darwin && cgo

package metal

import "unsafe"

/*
BufferRefFromDispatch unwraps a dispatch pointer into a Metal buffer handle.
*/
func BufferRefFromDispatch(pointer unsafe.Pointer) uintptr {
	return uintptr(unsafe.Pointer(resolveBufferRef(pointer)))
}
