//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <string.h>
#include "bridge_darwin.h"
*/
import "C"

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func metalTensorContents(value tensor.Tensor) unsafe.Pointer {
	target, ok := value.(*metalTensor)

	if !ok || target == nil {
		return nil
	}

	return metalBufferContents(unsafe.Pointer(target.buffer))
}

func metalBufferContents(buffer unsafe.Pointer) unsafe.Pointer {
	if buffer == nil {
		return nil
	}

	return C.metal_buffer_contents(C.MetalBufferRef(buffer))
}

func metalMemset(destination unsafe.Pointer, value byte, byteCount int) {
	if destination == nil || byteCount <= 0 {
		return
	}

	C.memset(destination, C.int(value), C.size_t(byteCount))
}
