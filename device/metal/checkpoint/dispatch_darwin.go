//go:build darwin && cgo

package checkpoint

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "core.h"

extern int metal_dispatch_checkpoint_encode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint64_t* dims,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_checkpoint_decode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t headerBytes,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
*/
import "C"

var errUnsupportedDType = errors.New("metal checkpoint: unsupported dtype")

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}

func DispatchCheckpointEncodeRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	rank, count uint32,
	dims []uint64,
) error {
	if count == 0 {
		return nil
	}

	var status C.MetalStatus
	var dimsPointer *C.uint64_t

	if len(dims) > 0 {
		dimsPointer = (*C.uint64_t)(unsafe.Pointer(&dims[0]))
	}

	code := C.metal_dispatch_checkpoint_encode_float32(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(rank),
		C.uint32_t(count),
		dimsPointer,
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchCheckpointDecodeRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	headerBytes, count uint32,
) error {
	if count == 0 {
		return nil
	}

	var status C.MetalStatus
	code := C.metal_dispatch_checkpoint_decode_float32(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(headerBytes),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func CheckpointElementCount(format dtype.DType, count int) (uint32, error) {
	switch format {
	case dtype.Float32:
		return uint32(count), nil
	default:
		return 0, errUnsupportedDType
	}
}

func CheckpointHeaderBytes(rank int) uint32 {
	return uint32(16 + rank*8)
}
