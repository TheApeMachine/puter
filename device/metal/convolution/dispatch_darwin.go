//go:build darwin && cgo

package convolution

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "conv2d.h"
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

func DispatchConv2DRefs(
	contextRef uintptr,
	inputBuffer uintptr,
	weightBuffer uintptr,
	biasBuffer uintptr,
	outputBuffer uintptr,
	format dtype.DType,
	batch uint32,
	inChannels uint32,
	inHeight uint32,
	inWidth uint32,
	outChannels uint32,
	kernelHeight uint32,
	kernelWidth uint32,
	outHeight uint32,
	outWidth uint32,
	strideHeight uint32,
	strideWidth uint32,
	paddingHeight uint32,
	paddingWidth uint32,
	dilationHeight uint32,
	dilationWidth uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_conv2d(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputBuffer)),
		C.MetalBufferRef(unsafe.Pointer(weightBuffer)),
		C.MetalBufferRef(unsafe.Pointer(biasBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		C.uint32_t(batch),
		C.uint32_t(inChannels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outChannels),
		C.uint32_t(kernelHeight),
		C.uint32_t(kernelWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.uint32_t(strideHeight),
		C.uint32_t(strideWidth),
		C.uint32_t(paddingHeight),
		C.uint32_t(paddingWidth),
		C.uint32_t(dilationHeight),
		C.uint32_t(dilationWidth),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

var errUnsupportedDType = errors.New("metal convolution: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
