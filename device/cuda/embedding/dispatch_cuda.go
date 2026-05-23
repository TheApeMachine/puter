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
*/
import "C"

//go:embed embedding.cuh
var embeddingHubSource string

//go:embed lookup.cu
var lookupDomainSource string

func moduleSource() string {
	parts := []string{
		embeddingHubSource,
		lookupDomainSource,
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
