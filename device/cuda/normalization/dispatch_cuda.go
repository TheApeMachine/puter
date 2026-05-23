//go:build cuda

package normalization

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
#include "normalization.h"
#include "groupnorm.h"
#include "instancenorm.h"
#include "batchnorm.h"
*/
import "C"

//go:embed normalization.cuh
var normalizationHubSource string

//go:embed groupnorm.cu
var groupnormDomainSource string

//go:embed instancenorm.cu
var instancenormDomainSource string

//go:embed batchnorm.cu
var batchnormDomainSource string

func moduleSource() string {
	parts := []string{
		normalizationHubSource,
		groupnormDomainSource,
		instancenormDomainSource,
		batchnormDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_normalization_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA normalization dtype"}

func DispatchGroupNorm(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	scaleBuffer C.CUDABufferRef,
	biasBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	config device.GroupNormConfig,
	batch uint32,
	channels uint32,
	spatial uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_groupnorm(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		C.uint32_t(config.Groups),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchInstanceNorm(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	scaleBuffer C.CUDABufferRef,
	biasBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	batch uint32,
	channels uint32,
	spatial uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_instancenorm(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchBatchNormEval(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	scaleBuffer C.CUDABufferRef,
	biasBuffer C.CUDABufferRef,
	meanBuffer C.CUDABufferRef,
	varianceBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	batch uint32,
	channels uint32,
	spatial uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_batchnorm_eval(
		contextRef,
		elementFormat,
		inputBuffer,
		scaleBuffer,
		biasBuffer,
		meanBuffer,
		varianceBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(channels),
		C.uint32_t(spatial),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchInstanceNormRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	scaleBuffer uintptr,
	biasBuffer uintptr,
	outputBuffer uintptr,
	batch uint32,
	channels uint32,
	spatial uint32,
	format dtype.DType,
) error {
	return DispatchInstanceNorm(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(inputBuffer)),
		C.CUDABufferRef(unsafe.Pointer(scaleBuffer)),
		C.CUDABufferRef(unsafe.Pointer(biasBuffer)),
		C.CUDABufferRef(unsafe.Pointer(outputBuffer)),
		batch,
		channels,
		spatial,
		format,
	)
}
