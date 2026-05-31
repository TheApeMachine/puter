//go:build darwin && cgo

package masking

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../internal/bridge/core.h"

extern int metal_dispatch_apply_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_causal_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_alibi_bias(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef scoresRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
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

func DispatchApplyMaskRefs(
	contextRef uintptr,
	inputRef, maskRef, outRef uintptr,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_apply_mask(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(maskRef)),
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

func DispatchCausalMaskRefs(
	contextRef uintptr,
	outRef uintptr,
	format dtype.DType,
	rows, cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_causal_mask(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
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

func DispatchALiBiBiasRefs(
	contextRef uintptr,
	scoresRef, slopeRef, outRef uintptr,
	format dtype.DType,
	rows, cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_alibi_bias(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(scoresRef)),
		C.MetalBufferRef(unsafe.Pointer(slopeRef)),
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

var errUnsupportedDType = errors.New("metal masking: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
