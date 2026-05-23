//go:build cuda

package sampling

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "sampling_dispatch.h"
*/
import "C"

//go:embed sampling.cuh
var samplingHubSource string

//go:embed sampling.cu
var samplingDomainSource string

func moduleSource() string {
	parts := []string{
		samplingHubSource,
		samplingDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_sampling_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA sampling dtype"}

func paddedCount(count uint32) uint32 {
	if count == 0 {
		return 0
	}

	padded := uint32(1)

	for padded < count {
		padded <<= 1
	}

	return padded
}

/*
PaddedCount returns the next power-of-two at least count for bitonic sort padding.
*/
func PaddedCount(count uint32) uint32 {
	return paddedCount(count)
}

/*
DispatchSampling launches greedy or nucleus sampling kernels on device.
*/
func DispatchSampling(
	contextRef C.CUDADeviceRef,
	operation C.int,
	logitsBuffer C.CUDABufferRef,
	scoresBuffer C.CUDABufferRef,
	indicesBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
	target float32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	padded := paddedCount(count)

	var status C.CUDAStatus
	code := C.cuda_dispatch_sampling(
		contextRef,
		operation,
		elementFormat,
		logitsBuffer,
		scoresBuffer,
		indicesBuffer,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(padded),
		C.float(target),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
