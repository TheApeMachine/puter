//go:build darwin && cgo

package normalization

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

func DispatchGroupNorm(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	scaleBuffer C.MetalBufferRef,
	biasBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	format dtype.DType,
	batch uint32,
	channels uint32,
	spatial uint32,
	groups uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_groupnorm(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		C.uint32_t(groups),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchGroupNormRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scaleBuffer uintptr,
	biasBuffer uintptr,
	outputBuffer uintptr,
	format dtype.DType,
	batch uint32,
	channels uint32,
	spatial uint32,
	groups uint32,
) error {
	return DispatchGroupNorm(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scaleBuffer)),
		C.MetalBufferRef(unsafe.Pointer(biasBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		format,
		batch,
		channels,
		spatial,
		groups,
	)
}

func DispatchModulatedLayerNorm(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	modulationBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
	rowsPerBatch uint32,
	modulationCols uint32,
	modulationSet uint32,
	epsilon float32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_modulated_layernorm(
		contextRef,
		elementFormat,
		inputBuffer,
		modulationBuffer,
		outputBuffer,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.uint32_t(rowsPerBatch),
		C.uint32_t(modulationCols),
		C.uint32_t(modulationSet),
		C.float(epsilon),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchModulatedLayerNormRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	modulationBuffer uintptr,
	outputBuffer uintptr,
	format dtype.DType,
	rows uint32,
	cols uint32,
	rowsPerBatch uint32,
	modulationCols uint32,
	modulationSet uint32,
	epsilon float32,
) error {
	return DispatchModulatedLayerNorm(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(modulationBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		format,
		rows,
		cols,
		rowsPerBatch,
		modulationCols,
		modulationSet,
		epsilon,
	)
}

var errUnsupportedDType = errors.New("metal normalization: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
