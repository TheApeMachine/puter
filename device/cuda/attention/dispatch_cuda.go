//go:build cuda

package attention

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "attention.h"
#include "scaled_dot_product.h"
*/
import "C"

//go:embed attention.cuh
var attentionHubSource string

//go:embed scaled_dot_product.cu
var scaledDotProductDomainSource string

func moduleSource() string {
	parts := []string{
		attentionHubSource,
		scaledDotProductDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_attention_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA attention dtype"}

func DispatchScaledDotProductAttention(
	contextRef C.CUDADeviceRef,
	queryBuffer C.CUDABufferRef,
	keyBuffer C.CUDABufferRef,
	valueBuffer C.CUDABufferRef,
	scoresBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	seqQ uint32,
	seqK uint32,
	depth uint32,
	valueDim uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_attention(
		contextRef,
		elementFormat,
		queryBuffer,
		keyBuffer,
		valueBuffer,
		scoresBuffer,
		outputBuffer,
		C.uint32_t(seqQ),
		C.uint32_t(seqK),
		C.uint32_t(depth),
		C.uint32_t(valueDim),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
