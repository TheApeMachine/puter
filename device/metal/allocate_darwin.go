//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <string.h>
#include "bridge_darwin.h"
*/
import "C"

import "unsafe"

func metalBufferContents(buffer uintptr) unsafe.Pointer {
	if buffer == 0 {
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
