//go:build darwin && cgo

package matmul

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "product.h"
*/
import "C"

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

func DispatchMatmul(
	contextRef C.MetalDeviceRef,
	leftBuffer C.MetalBufferRef,
	rightBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	rows uint32,
	inner uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_matmul(
		contextRef,
		elementFormat,
		leftBuffer,
		rightBuffer,
		outBuffer,
		C.uint32_t(rows),
		C.uint32_t(inner),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchMatmulRefs(
	contextRef uintptr,
	leftBuffer uintptr,
	rightBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	rows uint32,
	inner uint32,
	cols uint32,
) error {
	return DispatchMatmul(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(leftBuffer)),
		C.MetalBufferRef(unsafe.Pointer(rightBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		rows,
		inner,
		cols,
	)
}

var errUnsupportedDType = errors.New("metal matmul: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
