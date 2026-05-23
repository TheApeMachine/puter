//go:build cuda

package quant

import (
	_ "embed"
	"strings"
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

const (
	operationInt8Quant C.int = 2
)

func dispatchQuantization(
	contextRef C.CUDADeviceRef,
	operation C.int,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	scale float32,
	zeroPoint int32,
	count uint32,
) error {
	var status C.CUDAStatus
	code := C.cuda_dispatch_quantization(
		contextRef,
		operation,
		inputBuffer,
		outputBuffer,
		C.float(scale),
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
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	return dispatchQuantization(
		contextRef,
		operationInt8Quant,
		sourceBuffer,
		destinationBuffer,
		scale,
		int32(zeroPoint),
		count,
	)
}
