//go:build windows

package cpu

/*
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func platformAllocateAligned(byteCount int64) (unsafe.Pointer, error) {
	if byteCount <= 0 {
		return nil, nil
	}

	devicePointer := C._aligned_malloc(C.size_t(byteCount), C.size_t(workspaceAlign))

	if devicePointer == nil {
		return nil, tensor.ErrAllocatorExhausted
	}

	return unsafe.Pointer(devicePointer), nil
}

func platformRelease(devicePointer unsafe.Pointer) {
	if devicePointer == nil {
		return
	}

	C._aligned_free(devicePointer)
}
