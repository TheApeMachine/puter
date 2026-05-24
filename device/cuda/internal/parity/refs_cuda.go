//go:build cuda

package parity

/*
#cgo cuda CFLAGS: -I${SRCDIR}/../bridge
#cgo cuda LDFLAGS: -lcuda -lcudart -lpthread

#include "../bridge/core.h"
*/
import "C"

import "unsafe"

/*
DeviceRef converts a harness context reference for CUDA dispatch entry points.
*/
func DeviceRef(contextRef unsafe.Pointer) C.CUDADeviceRef {
	return C.CUDADeviceRef(contextRef)
}

/*
BufferRef converts a harness buffer reference for CUDA dispatch entry points.
*/
func BufferRef(bufferRef unsafe.Pointer) C.CUDABufferRef {
	return C.CUDABufferRef(bufferRef)
}
