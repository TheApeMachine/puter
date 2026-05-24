//go:build darwin && cgo

package random

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "normal.h"
*/
import "C"

/*
DispatchNormal launches the Metal random_normal_float32 kernel.
*/
func DispatchNormal(
	contextRef C.MetalDeviceRef,
	outBuffer C.MetalBufferRef,
	count uint32,
	seed uint64,
	counter uint64,
) error {
	if count == 0 {
		return nil
	}

	var status C.MetalStatus
	code := C.metal_dispatch_random_normal(
		contextRef,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(uint32(seed)),
		C.uint32_t(uint32(seed>>32)),
		C.uint32_t(uint32(counter)),
		C.uint32_t(uint32(counter>>32)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

/*
DispatchNormalRefs is the uintptr-friendly entry point used by the
backend Host implementation, which holds buffer handles as uintptrs.
*/
func DispatchNormalRefs(
	contextRef uintptr,
	outBuffer uintptr,
	count uint32,
	seed uint64,
	counter uint64,
) error {
	return DispatchNormal(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		count,
		seed,
		counter,
	)
}

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
