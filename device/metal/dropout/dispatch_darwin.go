//go:build darwin && cgo

package dropout

import (
	"errors"
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "mask.h"
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

func dropoutSeedState(seed uint64) (seedX, seedY, seedZ, seedW uint32) {
	seedX = uint32(seed)
	seedY = uint32(seed >> 32)
	seedZ = uint32(seed ^ 0x9e3779b9)
	seedW = uint32((seed >> 32) ^ 0x6c078965)

	return seedX, seedY, seedZ, seedW
}

func dropoutThreshold(keepProb float32) uint32 {
	return uint32(float64(keepProb) * (1 << 32))
}

func DispatchDropout(
	contextRef C.MetalDeviceRef,
	inputBuffer C.MetalBufferRef,
	outputBuffer C.MetalBufferRef,
	count uint32,
	config device.DropoutConfig,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if config.Rate < 0 || config.Rate > 1.0 {
		return errInvalidRate
	}

	if config.Rate == 0 {
		var status C.MetalStatus
		code := C.metal_dispatch_dropout(
			contextRef,
			elementFormat,
			inputBuffer,
			outputBuffer,
			C.uint32_t(count),
			1,
			0xFFFFFFFF,
			0, 0, 0, 0,
			0,
			&status,
		)

		if code != 0 {
			return metalStatusError(status)
		}

		return nil
	}

	keepProb := float32(1.0 - config.Rate)
	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)
	seedX, seedY, seedZ, seedW := dropoutSeedState(config.Seed)

	if math.IsInf(float64(scale), 1) {
		return errInvalidRate
	}

	var status C.MetalStatus
	code := C.metal_dispatch_dropout(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.uint32_t(count),
		C.float(scale),
		C.uint32_t(threshold),
		C.uint32_t(seedX),
		C.uint32_t(seedY),
		C.uint32_t(seedZ),
		C.uint32_t(seedW),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchDropoutRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	outputBuffer uintptr,
	count uint32,
	config device.DropoutConfig,
	format dtype.DType,
) error {
	return DispatchDropout(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		count,
		config,
		format,
	)
}

var (
	errUnsupportedDType = errors.New("metal dropout: unsupported dtype")
	errInvalidRate      = errors.New("metal dropout: invalid rate")
)

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
