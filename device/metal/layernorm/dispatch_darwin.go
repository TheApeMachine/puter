//go:build darwin && cgo

package layernorm

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "dispatch.h"
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

func DispatchLayerNorm(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	scaleBuffer C.MetalBufferRef,
	biasBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_layernorm(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		outputBuffer,
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

func DispatchLayerNormRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scaleBuffer uintptr,
	biasBuffer uintptr,
	outputBuffer uintptr,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	return DispatchLayerNorm(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scaleBuffer)),
		C.MetalBufferRef(unsafe.Pointer(biasBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		format,
		rows,
		cols,
	)
}

func DispatchLayerNormStatsRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	statsBuffer uintptr,
	rows uint32,
	cols uint32,
) error {
	return DispatchLayerNormStats(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(statsBuffer)),
		rows,
		cols,
	)
}

func DispatchLayerNormStats(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	statsBuffer C.MetalBufferRef,
	rows uint32,
	cols uint32,
) error {
	var status C.MetalStatus
	code := C.metal_dispatch_layernorm_stats(
		contextRef,
		inputBuffer,
		statsBuffer,
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

func DispatchLayerNormApplyRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scaleBuffer uintptr,
	biasBuffer uintptr,
	outputBuffer uintptr,
	statsBuffer uintptr,
	rows uint32,
	cols uint32,
) error {
	return DispatchLayerNormApply(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scaleBuffer)),
		C.MetalBufferRef(unsafe.Pointer(biasBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(statsBuffer)),
		rows,
		cols,
	)
}

func DispatchLayerNormApply(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	scaleBuffer C.MetalBufferRef,
	biasBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	statsBuffer C.MetalBufferRef,
	rows uint32,
	cols uint32,
) error {
	var status C.MetalStatus
	code := C.metal_dispatch_layernorm_apply(
		contextRef,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		outputBuffer,
		statsBuffer,
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

/*
DispatchRMSNorm launches the per-dtype RMSNorm kernel
(rmsnorm_<dtype>) declared in layer.metal. RMSNorm has no bias, so the
buffer binding is one short relative to LayerNorm: input, scale, out,
cols. See native/layer.m::metal_dispatch_rmsnorm for the encoder shape.
*/
func DispatchRMSNorm(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	scaleBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_layernorm_rmsnorm(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		outputBuffer,
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

/*
DispatchRMSNormRefs is the uintptr entry point used by the runtime
dispatcher. Mirrors DispatchLayerNormRefs.
*/
func DispatchRMSNormRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scaleBuffer uintptr,
	outputBuffer uintptr,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	return DispatchRMSNorm(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scaleBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		format,
		rows,
		cols,
	)
}

var errUnsupportedDType = errors.New("metal layernorm: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
