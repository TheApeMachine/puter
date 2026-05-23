//go:build cuda

package dot

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "dot.h"
#include "inner_product.h"
*/
import "C"

//go:embed dot.cuh
var dotHubSource string

//go:embed inner_product.cu
var innerProductDomainSource string

func moduleSource() string {
	parts := []string{
		dotHubSource,
		innerProductDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_dot_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA dot dtype"}

func DispatchDot(
	contextRef C.CUDADeviceRef,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
	scratchBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	partialCount := (count + 255) / 256

	var status C.CUDAStatus
	code := C.cuda_dispatch_dot(
		contextRef,
		elementFormat,
		leftBuffer,
		rightBuffer,
		scratchBuffer,
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

func DispatchDotRefs(
	contextRef uintptr,
	leftBuffer uintptr,
	rightBuffer uintptr,
	scratchBuffer uintptr,
	outBuffer uintptr,
	format dtype.DType,
	count uint32,
) error {
	return DispatchDot(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(leftBuffer)),
		C.CUDABufferRef(unsafe.Pointer(rightBuffer)),
		C.CUDABufferRef(unsafe.Pointer(scratchBuffer)),
		C.CUDABufferRef(unsafe.Pointer(outBuffer)),
		format,
		count,
	)
}
