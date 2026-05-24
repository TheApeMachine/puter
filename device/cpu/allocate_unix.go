//go:build unix

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

	var devicePointer unsafe.Pointer

	if C.posix_memalign(&devicePointer, C.size_t(workspaceAlign), C.size_t(byteCount)) != 0 {
		return nil, tensor.ErrAllocatorExhausted
	}

	return devicePointer, nil
}

func platformRelease(devicePointer unsafe.Pointer) {
	if devicePointer == nil {
		return
	}

	C.free(devicePointer)
}
