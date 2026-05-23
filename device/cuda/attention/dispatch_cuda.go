//go:build cuda

package attention

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
#include "attention.h"
#include "scaled_dot_product.h"
#include "flash.h"
#include "multihead.h"
#include "masking.h"
*/
import "C"

//go:embed attention.cuh
var attentionHubSource string

//go:embed scaled_dot_product.cu
var scaledDotProductDomainSource string

//go:embed flash.cu
var flashDomainSource string

//go:embed multihead.cu
var multiheadDomainSource string

//go:embed masking.cu
var maskingDomainSource string

func moduleSource() string {
	parts := []string{
		attentionHubSource,
		scaledDotProductDomainSource,
		flashDomainSource,
		multiheadDomainSource,
		maskingDomainSource,
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

func attentionVariant(config device.MultiHeadAttentionConfig) C.int {
	if config.WindowSize > 0 {
		return 2
	}

	kvHeads := config.KVHeadCount

	if kvHeads == 0 {
		kvHeads = config.NumHeads
	}

	if kvHeads < config.NumHeads {
		return 1
	}

	return 0
}

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

func DispatchFlashAttention(
	contextRef C.CUDADeviceRef,
	queryBuffer C.CUDABufferRef,
	keyBuffer C.CUDABufferRef,
	valueBuffer C.CUDABufferRef,
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
	code := C.cuda_dispatch_flash_attention(
		contextRef,
		elementFormat,
		queryBuffer,
		keyBuffer,
		valueBuffer,
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

func DispatchMultiHeadAttention(
	contextRef C.CUDADeviceRef,
	queryBuffer C.CUDABufferRef,
	keyBuffer C.CUDABufferRef,
	valueBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	config device.MultiHeadAttentionConfig,
	seqQ uint32,
	seqK uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	kvHeads := config.KVHeadCount

	if kvHeads == 0 {
		kvHeads = config.NumHeads
	}

	causal := C.uint32_t(0)

	if config.Causal {
		causal = 1
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_multi_head_attention(
		contextRef,
		elementFormat,
		attentionVariant(config),
		queryBuffer,
		keyBuffer,
		valueBuffer,
		outputBuffer,
		C.uint32_t(seqQ),
		C.uint32_t(seqK),
		C.uint32_t(config.NumHeads),
		C.uint32_t(kvHeads),
		C.uint32_t(config.HeadDim),
		C.uint32_t(config.WindowSize),
		causal,
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchApplyMask(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	maskBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	count uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_apply_mask(
		contextRef,
		elementFormat,
		inputBuffer,
		maskBuffer,
		outputBuffer,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchCausalMask(
	contextRef C.CUDADeviceRef,
	outputBuffer C.CUDABufferRef,
	rows uint32,
	cols uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_causal_mask(
		contextRef,
		elementFormat,
		outputBuffer,
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchALiBiBias(
	contextRef C.CUDADeviceRef,
	scoresBuffer C.CUDABufferRef,
	slopeBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	rows uint32,
	cols uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_alibi_bias(
		contextRef,
		elementFormat,
		scoresBuffer,
		slopeBuffer,
		outputBuffer,
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchApplyMaskRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	maskBuffer uintptr,
	outputBuffer uintptr,
	count uint32,
	format dtype.DType,
) error {
	return DispatchApplyMask(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(inputBuffer)),
		C.CUDABufferRef(unsafe.Pointer(maskBuffer)),
		C.CUDABufferRef(unsafe.Pointer(outputBuffer)),
		count,
		format,
	)
}
