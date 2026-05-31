//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/internal/bridge -I${SRCDIR}/internal/runtime
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "internal/bridge/core_private.h"

int metal_dispatch_page_write(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef storageRef,
    MetalBufferRef valuesRef,
    MetalBufferRef pageIDsRef,
    MetalBufferRef offsetsRef,
    MetalBufferRef outRef,
    uint32_t pageCount,
    uint32_t pageSize,
    uint32_t inner,
    uint32_t valueRows,
    uint32_t storageOffset,
    uint32_t outOffset,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_page_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef storageRef,
    MetalBufferRef pageTableRef,
    MetalBufferRef outRef,
    uint32_t pageCount,
    uint32_t pageSize,
    uint32_t inner,
    uint32_t outRows,
    uint32_t storageOffset,
    uint32_t outOffset,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_concat_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_concat_last_dim_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftRowBytes,
    uint32_t rightRowBytes,
    uint32_t rowBytes,
    uint32_t totalBytes,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_slice_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t sliceLen,
    uint32_t inputDimSize,
    uint32_t innerBytes,
    uint32_t start,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_transpose(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint32_t* permutation,
    const uint32_t* inputStrides,
    const uint32_t* outStrides,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_last_token_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seq,
    uint32_t hiddenBytes,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_upsample_nearest2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t outElements,
    uint64_t completionToken,
    MetalStatus* status
);
*/
import "C"

func (backend *Backend) KVPageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageCount, pageSize, inner, valueRows, storageOffset int,
	format dtype.DType,
) {
	if pageCount == 0 || pageSize == 0 || inner == 0 || valueRows == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_page_write(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(storage),
		resolveBufferRef(values),
		resolveBufferRef(pageIDs),
		resolveBufferRef(offsets),
		resolveBufferRef(output),
		C.uint32_t(pageCount),
		C.uint32_t(pageSize),
		C.uint32_t(inner),
		C.uint32_t(valueRows),
		C.uint32_t(storageOffset),
		C.uint32_t(storageOffset),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) KVPageGather(
	storage, pageTable, output unsafe.Pointer,
	pageCount, pageSize, inner, outRows, storageOffset int,
	format dtype.DType,
) {
	if pageCount == 0 || pageSize == 0 || inner == 0 || outRows == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_page_gather(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(storage),
		resolveBufferRef(pageTable),
		resolveBufferRef(output),
		C.uint32_t(pageCount),
		C.uint32_t(pageSize),
		C.uint32_t(inner),
		C.uint32_t(outRows),
		C.uint32_t(storageOffset),
		0,
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicConcat(
	left, right, output unsafe.Pointer,
	leftBytes, rightBytes int,
	format dtype.DType,
) {
	if leftBytes == 0 && rightBytes == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_concat_bytes(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(left),
		resolveBufferRef(right),
		resolveBufferRef(output),
		C.uint32_t(leftBytes),
		C.uint32_t(rightBytes),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicConcatLastDim(
	left, right, output unsafe.Pointer,
	leftRowBytes, rightRowBytes, rowBytes, totalBytes int,
	format dtype.DType,
) {
	if totalBytes == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_concat_last_dim_bytes(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(left),
		resolveBufferRef(right),
		resolveBufferRef(output),
		C.uint32_t(leftRowBytes),
		C.uint32_t(rightRowBytes),
		C.uint32_t(rowBytes),
		C.uint32_t(totalBytes),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicSlice(
	input, output unsafe.Pointer,
	sliceLen, inputDimSize, innerBytes, start, outBytes int,
	format dtype.DType,
) {
	if sliceLen == 0 || inputDimSize == 0 || innerBytes == 0 || outBytes == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_slice_bytes(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(input),
		resolveBufferRef(output),
		C.uint32_t(sliceLen),
		C.uint32_t(inputDimSize),
		C.uint32_t(innerBytes),
		C.uint32_t(start),
		C.uint32_t(outBytes),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicTranspose(
	input, output unsafe.Pointer,
	rank, count int,
	permutation, inputStrides, outputStrides []uint32,
	format dtype.DType,
) {
	if rank == 0 || count == 0 {
		return
	}

	if len(permutation) != rank || len(inputStrides) != rank || len(outputStrides) != rank {
		panic(tensor.ErrShapeMismatch)
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_transpose(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(input),
		resolveBufferRef(output),
		C.uint32_t(rank),
		C.uint32_t(count),
		(*C.uint32_t)(unsafe.Pointer(&permutation[0])),
		(*C.uint32_t)(unsafe.Pointer(&inputStrides[0])),
		(*C.uint32_t)(unsafe.Pointer(&outputStrides[0])),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicLastToken(
	input, output unsafe.Pointer,
	seq, hiddenBytes, outBytes int,
	format dtype.DType,
) {
	if seq == 0 || hiddenBytes == 0 || outBytes == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_last_token_bytes(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(input),
		resolveBufferRef(output),
		C.uint32_t(seq),
		C.uint32_t(hiddenBytes),
		C.uint32_t(outBytes),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) IntrinsicUpsampleNearest2D(
	input, output unsafe.Pointer,
	channels, inHeight, inWidth, outHeight, outWidth, outElements int,
	format dtype.DType,
) {
	if channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || outElements == 0 {
		return
	}

	elementFormat := metalElementDType(format)

	if elementFormat < 0 || backend == nil || backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	var status C.MetalStatus
	code := C.metal_dispatch_upsample_nearest2d_bytes(
		C.MetalDeviceRef(unsafe.Pointer(backend.bridge.device)),
		elementFormat,
		resolveBufferRef(input),
		resolveBufferRef(output),
		C.uint32_t(channels),
		C.uint32_t(inHeight),
		C.uint32_t(inWidth),
		C.uint32_t(outHeight),
		C.uint32_t(outWidth),
		C.uint32_t(outElements),
		0,
		&status,
	)

	if code != 0 {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func metalElementDType(format dtype.DType) C.int {
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
