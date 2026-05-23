//go:build cuda

package quant

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "quant_dispatch.h"
*/
import "C"

//go:embed quant.cuh
var quantHubSource string

//go:embed int8.cu
var int8DomainSource string

func moduleSource() string {
	parts := []string{
		quantHubSource,
		int8DomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_quant_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA quant dtype"}

func dispatchQuantization(
	contextRef C.CUDADeviceRef,
	srcFormat dtype.DType,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	elementFormat := elementDType(srcFormat)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if scale == 0 {
		return &dispatchError{code: -6, message: "cuda quant: zero scale"}
	}

	invScale := float32(1.0 / scale)

	var status C.CUDAStatus
	code := C.cuda_dispatch_quantization(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.float(invScale),
		C.int(zeroPoint),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchQuant(
	contextRef C.CUDADeviceRef,
	sourceBuffer C.CUDABufferRef,
	destinationBuffer C.CUDABufferRef,
	srcFormat dtype.DType,
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	return dispatchQuantization(
		contextRef,
		srcFormat,
		sourceBuffer,
		destinationBuffer,
		scale,
		zeroPoint,
		count,
	)
}

func DispatchQuantRefs(
	contextRef uintptr,
	sourceBuffer uintptr,
	destinationBuffer uintptr,
	srcFormat dtype.DType,
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	return DispatchQuant(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(sourceBuffer)),
		C.CUDABufferRef(unsafe.Pointer(destinationBuffer)),
		srcFormat,
		scale,
		zeroPoint,
		count,
	)
}
