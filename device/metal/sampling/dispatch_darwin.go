//go:build darwin && cgo

package sampling

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

/*
OperationGreedy selects argmax sampling (no scratch buffers).
OperationProbabilistic selects nucleus/top-k sampling with scratch buffers.
*/
const (
	OperationGreedy        = 0
	OperationProbabilistic = 1
)

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

func paddedCount(count uint32) uint32 {
	if count == 0 {
		return 0
	}

	padded := uint32(1)

	for padded < count {
		padded <<= 1
	}

	return padded
}

/*
PaddedCount returns the next power-of-two at least count for bitonic sort padding.
*/
func PaddedCount(count uint32) uint32 {
	return paddedCount(count)
}

func DispatchSampling(
	contextRef C.MetalDeviceRef,
	operation int,
	logitsBuffer C.MetalBufferRef,
	scoresBuffer C.MetalBufferRef,
	indicesBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	count uint32,
	target float32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	padded := paddedCount(count)

	var status C.MetalStatus
	code := C.metal_dispatch_sampling(
		contextRef,
		C.int(operation),
		elementFormat,
		logitsBuffer,
		scoresBuffer,
		indicesBuffer,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(padded),
		C.float(target),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchSamplingRefs(
	contextRef uintptr,
	operation int,
	logitsBuffer uintptr,
	scoresBuffer uintptr,
	indicesBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	count uint32,
	target float32,
) error {
	return DispatchSampling(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		operation,
		C.MetalBufferRef(unsafe.Pointer(logitsBuffer)),
		bufferRefOrNil(scoresBuffer),
		bufferRefOrNil(indicesBuffer),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		count,
		target,
	)
}

func bufferRefOrNil(bufferRef uintptr) C.MetalBufferRef {
	if bufferRef == 0 {
		return nil
	}

	return C.MetalBufferRef(unsafe.Pointer(bufferRef))
}

var errUnsupportedDType = errors.New("metal sampling: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
