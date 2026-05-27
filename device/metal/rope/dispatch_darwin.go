//go:build darwin && cgo

package rope

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "rotate.h"
*/
import "C"

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.MetalElementDTypeFloat32
	case dtype.Float16:
		return C.MetalElementDTypeFloat16
	case dtype.BFloat16:
		return C.MetalElementDTypeBFloat16
	default:
		return -1
	}
}

func DispatchRoPE(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
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

	halfMode, err := ropeHalfMode(config)

	if err != nil {
		return err
	}

	ropeFactor, lowFreqFactor, highFreqFactor, originalContext, err := ropeScalingParams(config)

	if err != nil {
		return err
	}

	pairCount := seqLen * numHeads * halfDim
	theta := C.float(config.BaseFreq)

	var status C.MetalStatus
	code := C.metal_dispatch_rope(
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
		return metalStatusError(status)
	}

	return nil
}

func DispatchRoPERefs(
	contextRef uintptr,
	inputBuffer uintptr,
	outputBuffer uintptr,
	config device.RoPEConfig,
	seqLen uint32,
	numHeads uint32,
	headDim uint32,
	format dtype.DType,
) error {
	return DispatchRoPE(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		config,
		seqLen,
		numHeads,
		headDim,
		format,
	)
}

func DispatchMultiAxisRoPE(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	config device.MultiAxisRoPEConfig,
	batch uint32,
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

	if batch == 0 || seqLen == 0 || numHeads == 0 || halfDim == 0 {
		return nil
	}

	pairCount := batch * seqLen * numHeads * halfDim
	theta := C.float(config.BaseFreq)

	var status C.MetalStatus
	code := C.metal_dispatch_multi_axis_rope(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.uint32_t(batch),
		C.uint32_t(seqLen),
		C.uint32_t(numHeads),
		C.uint32_t(headDim),
		C.uint32_t(pairCount),
		C.uint32_t(config.LatentSeqLen),
		C.uint32_t(config.LatentSide),
		theta,
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchMultiAxisRoPERefs(
	contextRef uintptr,
	inputBuffer uintptr,
	outputBuffer uintptr,
	config device.MultiAxisRoPEConfig,
	batch uint32,
	seqLen uint32,
	numHeads uint32,
	headDim uint32,
	format dtype.DType,
) error {
	return DispatchMultiAxisRoPE(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		config,
		batch,
		seqLen,
		numHeads,
		headDim,
		format,
	)
}

func ropeHalfMode(config device.RoPEConfig) (uint32, error) {
	switch config.Mode {
	case device.RoPEModeInterleaved:
		return 0, nil
	case device.RoPEModeHalf:
		return 1, nil
	default:
		return 0, errors.New("metal rope: unsupported mode")
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
		return 0, 0, 0, 0, errors.New("metal rope: unsupported scaling")
	}

	return C.float(config.ScalingFactor),
		C.float(config.LowFreqFactor),
		C.float(config.HighFreqFactor),
		C.uint32_t(config.OriginalContext),
		nil
}

var errUnsupportedDType = errors.New("metal rope: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
