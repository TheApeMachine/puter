//go:build cuda

package convolution

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "convolution.h"
#include "conv2d.h"
*/
import "C"

//go:embed convolution.cuh
var convolutionHubSource string

//go:embed conv2d.cu
var conv2dDomainSource string

func moduleSource() string {
	parts := []string{
		convolutionHubSource,
		conv2dDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_convolution_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA convolution dtype"}

func DispatchConv2D(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	weightBuffer C.CUDABufferRef,
	biasBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	batch uint32,
	inChannels uint32,
	inHeight uint32,
	inWidth uint32,
	outChannels uint32,
	kernelHeight uint32,
	kernelWidth uint32,
	outHeight uint32,
	outWidth uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_conv2d(
		contextRef,
		elementFormat,
		inputBuffer,
		weightBuffer,
		biasBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(inChannels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outChannels),
		C.uint32_t(kernelHeight),
		C.uint32_t(kernelWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
