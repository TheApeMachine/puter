//go:build darwin && cgo

package math

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../causal -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../causal/matrix.h"
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

func DispatchInvSqrtDimScaleRefs(
	contextRef uintptr,
	inputRef, dimRef, outRef uintptr,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_inv_sqrt_dim_scale(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(dimRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchLogSumExpRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	rows, cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_logsumexp(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchOuterRefs(
	contextRef uintptr,
	leftRef, rightRef, outRef uintptr,
	format dtype.DType,
	rows, cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_outer(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(leftRef)),
		C.MetalBufferRef(unsafe.Pointer(rightRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

var errUnsupportedDType = errors.New("metal math: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
