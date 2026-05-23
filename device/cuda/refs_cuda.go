//go:build cuda

package cuda

/*
#cgo cuda CFLAGS: -I${SRCDIR}/internal/bridge
#cgo cuda LDFLAGS: -lcuda -lcudart -lpthread

#include "internal/bridge/core.h"
*/
import "C"

import "unsafe"

/*
DeviceRef converts a harness context reference for CUDA dispatch entry points.
*/
func DeviceRef(contextRef uintptr) C.CUDADeviceRef {
	return C.CUDADeviceRef(unsafe.Pointer(contextRef))
}

/*
BufferRef converts a harness buffer reference for CUDA dispatch entry points.
*/
func BufferRef(bufferRef uintptr) C.CUDABufferRef {
	return C.CUDABufferRef(unsafe.Pointer(bufferRef))
}
