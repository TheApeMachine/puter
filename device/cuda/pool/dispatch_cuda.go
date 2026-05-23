//go:build cuda

package pool

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "maxpool.h"
#include "avgpool.h"
#include "adaptive.h"
*/
import "C"

//go:embed pool.cuh
var poolHubSource string

//go:embed maxpool.cu
var maxpoolDomainSource string

//go:embed avgpool.cu
var avgpoolDomainSource string

//go:embed adaptive.cu
var adaptiveDomainSource string

func moduleSource() string {
	parts := []string{
		poolHubSource,
		maxpoolDomainSource,
		avgpoolDomainSource,
		adaptiveDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_vision_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA pool dtype"}

/*
DispatchMaxPool2D launches the CUDA max-pool2d kernel for the given dtype.
*/
func DispatchMaxPool2D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	batch uint32,
	channels uint32,
	inHeight uint32,
	inWidth uint32,
	outHeight uint32,
	outWidth uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_max_pool2d(
		contextRef,
		elementFormat,
		inputRef,
		outputRef,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchAvgPool2D launches the CUDA avg-pool2d kernel for the given dtype.
*/
func DispatchAvgPool2D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	batch uint32,
	channels uint32,
	inHeight uint32,
	inWidth uint32,
	outHeight uint32,
	outWidth uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_avg_pool2d(
		contextRef,
		elementFormat,
		inputRef,
		outputRef,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchAdaptiveMaxPool2D launches the CUDA adaptive max-pool2d kernel.
*/
func DispatchAdaptiveMaxPool2D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	batch uint32,
	channels uint32,
	inHeight uint32,
	inWidth uint32,
	outHeight uint32,
	outWidth uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_adaptive_max_pool2d(
		contextRef,
		elementFormat,
		inputRef,
		outputRef,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchAdaptiveAvgPool2D launches the CUDA adaptive avg-pool2d kernel.
*/
func DispatchAdaptiveAvgPool2D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	batch uint32,
	channels uint32,
	inHeight uint32,
	inWidth uint32,
	outHeight uint32,
	outWidth uint32,
	completionToken uint64,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_adaptive_avg_pool2d(
		contextRef,
		elementFormat,
		inputRef,
		outputRef,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.ulonglong(completionToken),
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
