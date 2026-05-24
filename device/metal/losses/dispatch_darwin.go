//go:build darwin && cgo

package losses

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

func pairLossOperation(kernel LossKernel) C.int {
	switch kernel {
	case KernelMSE:
		return 0
	case KernelMAE:
		return 1
	case KernelHuber:
		return 2
	case KernelBinaryCrossEntropy:
		return 3
	case KernelKLDivergence:
		return 4
	default:
		return -1
	}
}

func partialCount(count uint32) uint32 {
	return (count + 255) / 256
}

func DispatchPairLoss(
	contextRef C.MetalDeviceRef,
	predictionsBuffer C.MetalBufferRef,
	targetsBuffer C.MetalBufferRef,
	scratchBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	kernel LossKernel,
	count uint32,
) error {
	elementFormat := elementDType(format)
	operation := pairLossOperation(kernel)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if operation < 0 {
		return errUnsupportedKernel
	}

	partials := partialCount(count)

	var status C.MetalStatus
	code := C.metal_dispatch_pair_loss(
		contextRef,
		operation,
		elementFormat,
		predictionsBuffer,
		targetsBuffer,
		scratchBuffer,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(partials),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchCrossEntropy(
	contextRef C.MetalDeviceRef,
	logitsBuffer C.MetalBufferRef,
	targetsBuffer C.MetalBufferRef,
	scratchBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	batch uint32,
	classes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_cross_entropy_loss(
		contextRef,
		elementFormat,
		logitsBuffer,
		targetsBuffer,
		scratchBuffer,
		outBuffer,
		C.uint32_t(batch),
		C.uint32_t(classes),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchPairLossRefs(
	contextRef uintptr,
	predictionsBuffer uintptr,
	targetsBuffer uintptr,
	scratchBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	kernel LossKernel,
	count uint32,
) error {
	return DispatchPairLoss(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(predictionsBuffer)),
		C.MetalBufferRef(unsafe.Pointer(targetsBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scratchBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		kernel,
		count,
	)
}

func DispatchCrossEntropyRefs(
	contextRef uintptr,
	logitsBuffer uintptr,
	targetsBuffer uintptr,
	scratchBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	batch uint32,
	classes uint32,
) error {
	return DispatchCrossEntropy(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(logitsBuffer)),
		C.MetalBufferRef(unsafe.Pointer(targetsBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scratchBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		batch,
		classes,
	)
}

var (
	errUnsupportedDType  = errors.New("metal losses: unsupported dtype")
	errUnsupportedKernel = errors.New("metal losses: unsupported kernel")
)

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
