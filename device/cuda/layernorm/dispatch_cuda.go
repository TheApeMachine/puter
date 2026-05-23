//go:build cuda

package layernorm

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "native/layer.h"
*/
import "C"

//go:embed layernorm.cuh
var layernormHubSource string

//go:embed layer.cu
var layerDomainSource string

func moduleSource() string {
	parts := []string{
		layernormHubSource,
		layerDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_layernorm_register_module_source(source)
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

/*
DispatchLayerNorm launches the CUDA layernorm kernel for the given dtype.
*/
func DispatchLayerNorm(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	scaleRef C.CUDABufferRef,
	biasRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_layernorm(
		contextRef,
		elementFormat,
		inputRef,
		scaleRef,
		biasRef,
		outputRef,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchRMSNorm launches the CUDA rmsnorm kernel for the given dtype.
*/
func DispatchRMSNorm(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	scaleRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_rmsnorm(
		contextRef,
		elementFormat,
		inputRef,
		scaleRef,
		outputRef,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA layernorm dtype"}

func DispatchLayerNormRefs(
	contextRef uintptr,
	inputRef uintptr,
	scaleRef uintptr,
	biasRef uintptr,
	outputRef uintptr,
	format dtype.DType,
	rows uint32,
	cols uint32,
	completionToken uint64,
) error {
	return DispatchLayerNorm(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(inputRef)),
		C.CUDABufferRef(unsafe.Pointer(scaleRef)),
		C.CUDABufferRef(unsafe.Pointer(biasRef)),
		C.CUDABufferRef(unsafe.Pointer(outputRef)),
		format,
		rows,
		cols,
		completionToken,
	)
}
