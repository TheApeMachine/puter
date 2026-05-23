//go:build cuda

package embedding

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "embedding.h"
#include "lookup.h"
#include "bag.h"
#include "timestep.h"
*/
import "C"

//go:embed embedding.cuh
var embeddingHubSource string

//go:embed lookup.cu
var lookupDomainSource string

//go:embed bag.cu
var bagDomainSource string

//go:embed timestep.cu
var timestepDomainSource string

func moduleSource() string {
	parts := []string{
		embeddingHubSource,
		lookupDomainSource,
		bagDomainSource,
		timestepDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_embedding_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA embedding dtype"}

func DispatchLookup(
	contextRef C.CUDADeviceRef,
	tableBuffer C.CUDABufferRef,
	indicesBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	errorFlagBuffer C.CUDABufferRef,
	format dtype.DType,
	vocab uint32,
	hidden uint32,
	indexCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_embedding_lookup(
		contextRef,
		elementFormat,
		tableBuffer,
		indicesBuffer,
		outputBuffer,
		errorFlagBuffer,
		C.uint32_t(vocab),
		C.uint32_t(hidden),
		C.uint32_t(indexCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchTimestepEmbedding(
	contextRef C.CUDADeviceRef,
	timestepsBuffer C.CUDABufferRef,
	maxPeriodBuffer C.CUDABufferRef,
	downscaleBuffer C.CUDABufferRef,
	flipBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
	dim uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_timestep_embedding(
		contextRef,
		elementFormat,
		timestepsBuffer,
		maxPeriodBuffer,
		downscaleBuffer,
		flipBuffer,
		outputBuffer,
		C.uint32_t(count),
		C.uint32_t(dim),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchBag(
	contextRef C.CUDADeviceRef,
	tableBuffer C.CUDABufferRef,
	indicesBuffer C.CUDABufferRef,
	offsetsBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	vocab uint32,
	hidden uint32,
	indexCount uint32,
	bagCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_embedding_bag(
		contextRef,
		elementFormat,
		tableBuffer,
		indicesBuffer,
		offsetsBuffer,
		outputBuffer,
		C.uint32_t(vocab),
		C.uint32_t(hidden),
		C.uint32_t(indexCount),
		C.uint32_t(bagCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchTimestepEmbedding(
	contextRef C.CUDADeviceRef,
	timestepsBuffer C.CUDABufferRef,
	maxPeriodBuffer C.CUDABufferRef,
	downscaleBuffer C.CUDABufferRef,
	flipBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
	dim uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_timestep_embedding(
		contextRef,
		elementFormat,
		timestepsBuffer,
		maxPeriodBuffer,
		downscaleBuffer,
		flipBuffer,
		outputBuffer,
		C.uint32_t(count),
		C.uint32_t(dim),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
