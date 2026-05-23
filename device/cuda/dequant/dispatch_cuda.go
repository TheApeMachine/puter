//go:build cuda

package dequant

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "dequant_dispatch.h"
*/
import "C"

//go:embed dequant.cuh
var dequantHubSource string

//go:embed int8.cu
var int8DomainSource string

func moduleSource() string {
	parts := []string{
		dequantHubSource,
		int8DomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_dequant_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA dequant dtype"}

const (
	operationInt8Dequant C.int = 0
	operationInt4Dequant C.int = 1
)

func dispatchDequantization(
	contextRef C.CUDADeviceRef,
	operation C.int,
	dstFormat dtype.DType,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	elementFormat := elementDType(dstFormat)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_dequantization(
		contextRef,
		operation,
		elementFormat,
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

func DispatchDequant(
	contextRef C.CUDADeviceRef,
	sourceBuffer C.CUDABufferRef,
	destinationBuffer C.CUDABufferRef,
	dstFormat dtype.DType,
	scale float32,
	zeroPoint int8,
	count uint32,
) error {
	return dispatchDequantization(
		contextRef,
		operationInt8Dequant,
		dstFormat,
		sourceBuffer,
		destinationBuffer,
		scale,
		zeroPoint,
		count,
	)
}

func DispatchDequant4(
	contextRef C.CUDADeviceRef,
	sourceBuffer C.CUDABufferRef,
	destinationBuffer C.CUDABufferRef,
	dstFormat dtype.DType,
	scale float32,
	zeroPoint int8,
	pairCount uint32,
) error {
	return dispatchDequantization(
		contextRef,
		operationInt4Dequant,
		dstFormat,
		sourceBuffer,
		destinationBuffer,
		scale,
		zeroPoint,
		pairCount,
	)
}
