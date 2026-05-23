//go:build cuda

package reduction

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "reduction.h"
#include "aggregate.h"
*/
import "C"

//go:embed reduction.cuh
var reductionHubSource string

//go:embed aggregate.cu
var aggregateDomainSource string

func moduleSource() string {
	parts := []string{
		reductionHubSource,
		aggregateDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_reduction_register_module_source(source)
}

func init() {
	registerModuleSource()
}

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.CUDAElementDTypeFloat32
	case dtype.Float16:
		return C.CUDAElementDTypeFloat16
	case dtype.BFloat16:
		return C.CUDAElementDTypeBFloat16
	case dtype.Float64:
		return C.CUDAElementDTypeFloat64
	case dtype.Float8E4M3:
		return C.CUDAElementDTypeFloat8E4M3
	case dtype.Float8E5M2:
		return C.CUDAElementDTypeFloat8E5M2
	default:
		return -1
	}
}

func kernelOperation(kernel ReductionKernel) C.int {
	switch kernel {
	case KernelSum:
		return 0
	case KernelProd:
		return 2
	case KernelMin:
		return 3
	case KernelMax:
		return 4
	case KernelL1Norm:
		return 7
	default:
		return -1
	}
}

func cudaStatusError(status C.CUDAStatus) error {
	if status.code == 0 {
		return nil
	}

	message := C.GoString(&status.message[0])
	return &dispatchError{code: int(status.code), message: message}
}

type dispatchError struct {
	code    int
	message string
}

func (dispatchError *dispatchError) Error() string {
	return dispatchError.message
}

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA reduction dtype"}

var errUnsupportedKernel = &dispatchError{code: -6, message: "unsupported CUDA reduction kernel"}

/*
DispatchReduction launches partial + finalize reduction kernels on device.
*/
func DispatchReduction(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	scratchABuffer C.CUDABufferRef,
	scratchBBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
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

	var status C.CUDAStatus
	code := C.cuda_dispatch_reduction(
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
		return cudaStatusError(status)
	}

	return nil
}
