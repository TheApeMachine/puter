//go:build darwin && cgo

package reduction

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "aggregate.h"
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
	case dtype.Float64:
		return C.MetalElementDTypeFloat64
	default:
		return -1
	}
}

/*
Reduction operation codes match aggregate.metal reduction_partial_a dispatch.
*/
const (
	metalReductionSum    C.int = 0
	metalReductionProd   C.int = 2
	metalReductionMin    C.int = 3
	metalReductionMax    C.int = 4
	metalReductionL1Norm C.int = 7
)

func kernelOperation(kernel ReductionKernel) C.int {
	switch kernel {
	case KernelSum:
		return metalReductionSum
	case KernelProd:
		return metalReductionProd
	case KernelMin:
		return metalReductionMin
	case KernelMax:
		return metalReductionMax
	case KernelL1Norm:
		return metalReductionL1Norm
	default:
		return -1
	}
}

func DispatchReduction(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	scratchABuffer C.MetalBufferRef,
	scratchBBuffer C.MetalBufferRef,
	outBuffer C.MetalBufferRef,
	format dtype.DType,
	kernel ReductionKernel,
	count uint32,
) error {
	elementFormat := elementDType(format)
	operation := kernelOperation(kernel)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if operation < 0 {
		return errUnsupportedKernel
	}

	partialCount := (count + 255) / 256

	var status C.MetalStatus
	code := C.metal_dispatch_reduction(
		contextRef,
		operation,
		elementFormat,
		inputBuffer,
		scratchABuffer,
		scratchBBuffer,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(partialCount),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchReductionRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scratchABuffer uintptr,
	scratchBBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	kernel ReductionKernel,
	count uint32,
) error {
	return DispatchReduction(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(scratchABuffer)),
		C.MetalBufferRef(unsafe.Pointer(scratchBBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outBuffer)),
		format,
		kernel,
		count,
	)
}

var (
	errUnsupportedDType  = errors.New("metal reduction: unsupported dtype")
	errUnsupportedKernel = errors.New("metal reduction: unsupported kernel")
)

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
