//go:build darwin && cgo

package attention

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

#include "multihead.h"
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

func DispatchMultiHeadAttentionRefs(
	contextRef uintptr,
	queryBuffer uintptr,
	keyBuffer uintptr,
	valueBuffer uintptr,
	outputBuffer uintptr,
	config device.MultiHeadAttentionConfig,
	seqQ uint32,
	seqK uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	variant := attentionVariant(config)
	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	var causal uint32

	if config.Causal {
		causal = 1
	}

	var status C.MetalStatus
	code := C.metal_dispatch_multi_head_attention(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.int(variant),
		C.MetalBufferRef(unsafe.Pointer(queryBuffer)),
		C.MetalBufferRef(unsafe.Pointer(keyBuffer)),
		C.MetalBufferRef(unsafe.Pointer(valueBuffer)),
		C.MetalBufferRef(unsafe.Pointer(outputBuffer)),
		C.uint32_t(seqQ),
		C.uint32_t(seqK),
		C.uint32_t(config.NumHeads),
		C.uint32_t(kvHeads),
		C.uint32_t(config.HeadDim),
		C.uint32_t(config.WindowSize),
		C.uint32_t(causal),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func attentionVariant(config device.MultiHeadAttentionConfig) int {
	if config.WindowSize > 0 {
		return 2
	}

	if config.KVHeadCount > 0 && config.KVHeadCount < config.NumHeads {
		return 1
	}

	return 0
}

var errUnsupportedDType = errors.New("metal attention: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
