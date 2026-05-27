//go:build cuda

package rope

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
#include "rope.h"
#include "rotate.h"
*/
import "C"

//go:embed rope.cuh
var ropeHubSource string

//go:embed rotate.cu
var rotateDomainSource string

func moduleSource() string {
	parts := []string{
		ropeHubSource,
		rotateDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_rope_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA rope dtype"}

func DispatchRoPE(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	config device.RoPEConfig,
	seqLen uint32,
	numHeads uint32,
	headDim uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if err := config.Validate(); err != nil {
		return err
	}

	halfDim := headDim / 2

	if halfDim == 0 {
		return nil
	}

	pairCount := seqLen * numHeads * halfDim
	theta := C.float(config.BaseFreq)
	halfMode, err := ropeHalfMode(config)

	if err != nil {
		return err
	}

	ropeFactor, lowFreqFactor, highFreqFactor, originalContext, err := ropeScalingParams(config)

	if err != nil {
		return err
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_rope(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.uint32_t(seqLen),
		C.uint32_t(numHeads),
		C.uint32_t(headDim),
		C.uint32_t(pairCount),
		theta,
		ropeFactor,
		lowFreqFactor,
		highFreqFactor,
		originalContext,
		C.uint32_t(halfMode),
		C.uint32_t(config.StartPosition),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func ropeHalfMode(config device.RoPEConfig) (uint32, error) {
	switch config.Mode {
	case device.RoPEModeInterleaved:
		return 0, nil
	case device.RoPEModeHalf:
		return 1, nil
	default:
		return 0, &dispatchError{code: -7, message: "unsupported CUDA rope mode"}
	}
}

func ropeScalingParams(config device.RoPEConfig) (
	C.float,
	C.float,
	C.float,
	C.uint32_t,
	error,
) {
	if config.Scaling == device.RoPEScalingNone {
		return 1.0, 0.0, 0.0, 0, nil
	}

	if config.Scaling != device.RoPEScalingLlama3 {
		return 0, 0, 0, 0, &dispatchError{code: -8, message: "unsupported CUDA rope scaling"}
	}

	return C.float(config.ScalingFactor),
		C.float(config.LowFreqFactor),
		C.float(config.HighFreqFactor),
		C.uint32_t(config.OriginalContext),
		nil
}

func DispatchRoPEPairs(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	cosBuffer C.CUDABufferRef,
	sinBuffer C.CUDABufferRef,
	halfDim uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if halfDim == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_rope_pairs(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		cosBuffer,
		sinBuffer,
		C.uint32_t(halfDim),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchRoPEPairsRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	outputBuffer uintptr,
	cosBuffer uintptr,
	sinBuffer uintptr,
	halfDim uint32,
	format dtype.DType,
) error {
	return DispatchRoPEPairs(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(inputBuffer)),
		C.CUDABufferRef(unsafe.Pointer(outputBuffer)),
		C.CUDABufferRef(unsafe.Pointer(cosBuffer)),
		C.CUDABufferRef(unsafe.Pointer(sinBuffer)),
		halfDim,
		format,
	)
}
