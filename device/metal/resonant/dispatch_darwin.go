//go:build darwin && cgo

package resonant

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "update.h"
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

func resonantUpdateParams(
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
) C.MetalResonantUpdateParams {
	elementCount := batchTime * headCount * headDim
	zeroDiag := uint32(0)

	if config.ZeroDiag {
		zeroDiag = 1
	}

	return C.MetalResonantUpdateParams{
		n:         C.uint32_t(elementCount),
		D:         C.uint32_t(headDim),
		H:         C.uint32_t(headCount),
		inv_D:     C.float(1.0 / float32(headDim)),
		scale:     C.float(config.Scale),
		damping:   C.float(config.Damping),
		zero_diag: C.uint32_t(zeroDiag),
	}
}

func DispatchResonantUpdateForward(
	contextRef C.MetalDeviceRef,
	xBuffer C.MetalBufferRef,
	yBuffer C.MetalBufferRef,
	vrBuffer C.MetalBufferRef,
	viBuffer C.MetalBufferRef,
	diagBuffer C.MetalBufferRef,
	xOutBuffer C.MetalBufferRef,
	yOutBuffer C.MetalBufferRef,
	aOutBuffer C.MetalBufferRef,
	bOutBuffer C.MetalBufferRef,
	invROutBuffer C.MetalBufferRef,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if batchTime*headCount*headDim == 0 {
		return nil
	}

	params := resonantUpdateParams(batchTime, headCount, headDim, config)
	var status C.MetalStatus
	code := C.metal_dispatch_resonant_update_forward(
		contextRef,
		elementFormat,
		xBuffer,
		yBuffer,
		vrBuffer,
		viBuffer,
		diagBuffer,
		xOutBuffer,
		yOutBuffer,
		aOutBuffer,
		bOutBuffer,
		invROutBuffer,
		&params,
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchResonantUpdateForwardRefs(
	contextRef uintptr,
	xBuffer uintptr,
	yBuffer uintptr,
	vrBuffer uintptr,
	viBuffer uintptr,
	diagBuffer uintptr,
	xOutBuffer uintptr,
	yOutBuffer uintptr,
	aOutBuffer uintptr,
	bOutBuffer uintptr,
	invROutBuffer uintptr,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) error {
	return DispatchResonantUpdateForward(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(xBuffer)),
		C.MetalBufferRef(unsafe.Pointer(yBuffer)),
		C.MetalBufferRef(unsafe.Pointer(vrBuffer)),
		C.MetalBufferRef(unsafe.Pointer(viBuffer)),
		C.MetalBufferRef(unsafe.Pointer(diagBuffer)),
		C.MetalBufferRef(unsafe.Pointer(xOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(yOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(aOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(bOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(invROutBuffer)),
		batchTime,
		headCount,
		headDim,
		config,
		format,
	)
}

func DispatchResonantUpdateBackward(
	contextRef C.MetalDeviceRef,
	gradXOutBuffer C.MetalBufferRef,
	gradYOutBuffer C.MetalBufferRef,
	xBuffer C.MetalBufferRef,
	yBuffer C.MetalBufferRef,
	diagBuffer C.MetalBufferRef,
	aBuffer C.MetalBufferRef,
	bBuffer C.MetalBufferRef,
	invRBuffer C.MetalBufferRef,
	gradXBuffer C.MetalBufferRef,
	gradYBuffer C.MetalBufferRef,
	gradVRBuffer C.MetalBufferRef,
	gradVIBuffer C.MetalBufferRef,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if batchTime*headCount*headDim == 0 {
		return nil
	}

	params := resonantUpdateParams(batchTime, headCount, headDim, config)
	var status C.MetalStatus
	code := C.metal_dispatch_resonant_update_backward(
		contextRef,
		elementFormat,
		gradXOutBuffer,
		gradYOutBuffer,
		xBuffer,
		yBuffer,
		diagBuffer,
		aBuffer,
		bBuffer,
		invRBuffer,
		gradXBuffer,
		gradYBuffer,
		gradVRBuffer,
		gradVIBuffer,
		&params,
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchResonantUpdateBackwardRefs(
	contextRef uintptr,
	gradXOutBuffer uintptr,
	gradYOutBuffer uintptr,
	xBuffer uintptr,
	yBuffer uintptr,
	diagBuffer uintptr,
	aBuffer uintptr,
	bBuffer uintptr,
	invRBuffer uintptr,
	gradXBuffer uintptr,
	gradYBuffer uintptr,
	gradVRBuffer uintptr,
	gradVIBuffer uintptr,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) error {
	return DispatchResonantUpdateBackward(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(gradXOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gradYOutBuffer)),
		C.MetalBufferRef(unsafe.Pointer(xBuffer)),
		C.MetalBufferRef(unsafe.Pointer(yBuffer)),
		C.MetalBufferRef(unsafe.Pointer(diagBuffer)),
		C.MetalBufferRef(unsafe.Pointer(aBuffer)),
		C.MetalBufferRef(unsafe.Pointer(bBuffer)),
		C.MetalBufferRef(unsafe.Pointer(invRBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gradXBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gradYBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gradVRBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gradVIBuffer)),
		batchTime,
		headCount,
		headDim,
		config,
		format,
	)
}

var errUnsupportedDType = errors.New("metal resonant: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
