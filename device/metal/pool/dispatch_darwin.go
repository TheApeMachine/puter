//go:build darwin && cgo

package pool

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "pool.h"
*/
import "C"

var errUnsupportedDType = errors.New("metal pool: unsupported dtype")

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.MetalElementDTypeFloat32
	case dtype.Float16:
		return C.MetalElementDTypeFloat16
	case dtype.BFloat16:
		return C.MetalElementDTypeBFloat16
	default:
		return -1
	}
}

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}

func DispatchPool2DRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	batch, channels, inHeight, inWidth, outHeight, outWidth uint32,
	useMax, adaptive bool,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_pool2d(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.bool(useMax),
		C.bool(adaptive),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}
