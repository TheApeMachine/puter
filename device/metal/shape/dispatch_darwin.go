//go:build darwin && cgo

package shape

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

extern int metal_dispatch_copy_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t byteCount,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_concat_bytes(
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

extern int metal_dispatch_split2_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_slice_bytes(
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

extern int metal_dispatch_last_token_bytes(
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

extern int metal_dispatch_transpose2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_upsample_nearest2d_bytes(
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

extern int metal_dispatch_merge_heads(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t seq,
    uint32_t heads,
    uint32_t headDim,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef sourceRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t sourceRows,
    uint32_t inner,
    uint32_t outRows,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_scatter(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef targetRef,
    MetalBufferRef indicesRef,
    MetalBufferRef updatesRef,
    MetalBufferRef outRef,
    uint32_t targetRows,
    uint32_t inner,
    uint32_t updateRows,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_page_write(
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

extern int metal_dispatch_page_gather(
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

extern int metal_dispatch_where(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef maskRef,
    MetalBufferRef positiveRef,
    MetalBufferRef negativeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_masked_fill(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef fillRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_transpose(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint32_t* permutation,
    const uint32_t* inputStrides,
    const uint32_t* outputStrides,
    uint64_t completionToken,
    MetalStatus* status
);
*/
import "C"

var errUnsupportedDType = errors.New("metal shape: unsupported dtype")

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

func elementByteSize(format dtype.DType) int {
	switch format {
	case dtype.Float32, dtype.Int32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 0
	}
}

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}

func DispatchCopyBytesRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	byteCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || byteCount == 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_copy_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(byteCount),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchConcatBytesRefs(
	contextRef uintptr,
	leftRef, rightRef, outRef uintptr,
	format dtype.DType,
	leftBytes, rightBytes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_concat_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(leftRef)),
		C.MetalBufferRef(unsafe.Pointer(rightRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(leftBytes),
		C.uint32_t(rightBytes),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchSplit2BytesRefs(
	contextRef uintptr,
	inputRef, leftRef, rightRef uintptr,
	format dtype.DType,
	leftBytes, rightBytes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_split2_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(leftRef)),
		C.MetalBufferRef(unsafe.Pointer(rightRef)),
		C.uint32_t(leftBytes),
		C.uint32_t(rightBytes),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchSliceBytesRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	sliceLen, inputDimSize, innerBytes, start, outBytes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_slice_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(sliceLen),
		C.uint32_t(inputDimSize),
		C.uint32_t(innerBytes),
		C.uint32_t(start),
		C.uint32_t(outBytes),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchLastTokenBytesRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	seq, hiddenBytes, outBytes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_last_token_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(seq),
		C.uint32_t(hiddenBytes),
		C.uint32_t(outBytes),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchTranspose2DBytesRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	rows, cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_transpose2d_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchUpsampleNearest2DBytesRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	channels, inHeight, inWidth, outHeight, outWidth, outElements uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_upsample_nearest2d_bytes(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
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
		return metalStatusError(status)
	}

	return nil
}

func DispatchMergeHeadsRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	batch, seq, heads, headDim uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_merge_heads(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(batch),
		C.uint32_t(seq),
		C.uint32_t(heads),
		C.uint32_t(headDim),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchGatherRefs(
	contextRef uintptr,
	sourceRef, indicesRef, outRef uintptr,
	format dtype.DType,
	sourceRows, inner, outRows uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_gather(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(sourceRef)),
		C.MetalBufferRef(unsafe.Pointer(indicesRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(sourceRows),
		C.uint32_t(inner),
		C.uint32_t(outRows),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchScatterRefs(
	contextRef uintptr,
	targetRef, indicesRef, updatesRef, outRef uintptr,
	format dtype.DType,
	targetRows, inner, updateRows uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_scatter(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(targetRef)),
		C.MetalBufferRef(unsafe.Pointer(indicesRef)),
		C.MetalBufferRef(unsafe.Pointer(updatesRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(targetRows),
		C.uint32_t(inner),
		C.uint32_t(updateRows),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchPageWriteRefs(
	contextRef uintptr,
	storageRef, valuesRef, pageIDsRef, offsetsRef, outRef uintptr,
	format dtype.DType,
	pageCount, pageSize, inner, valueRows, storageOffset, outOffset uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_page_write(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(storageRef)),
		C.MetalBufferRef(unsafe.Pointer(valuesRef)),
		C.MetalBufferRef(unsafe.Pointer(pageIDsRef)),
		C.MetalBufferRef(unsafe.Pointer(offsetsRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(pageCount),
		C.uint32_t(pageSize),
		C.uint32_t(inner),
		C.uint32_t(valueRows),
		C.uint32_t(storageOffset),
		C.uint32_t(outOffset),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchPageGatherRefs(
	contextRef uintptr,
	storageRef, pageTableRef, outRef uintptr,
	format dtype.DType,
	pageCount, pageSize, inner, outRows, storageOffset, outOffset uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_page_gather(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(storageRef)),
		C.MetalBufferRef(unsafe.Pointer(pageTableRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(pageCount),
		C.uint32_t(pageSize),
		C.uint32_t(inner),
		C.uint32_t(outRows),
		C.uint32_t(storageOffset),
		C.uint32_t(outOffset),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchWhereRefs(
	contextRef uintptr,
	maskRef, positiveRef, negativeRef, outRef uintptr,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_where(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(maskRef)),
		C.MetalBufferRef(unsafe.Pointer(positiveRef)),
		C.MetalBufferRef(unsafe.Pointer(negativeRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchMaskedFillRefs(
	contextRef uintptr,
	inputRef, maskRef, fillRef, outRef uintptr,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_masked_fill(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(maskRef)),
		C.MetalBufferRef(unsafe.Pointer(fillRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchTransposeRefs(
	contextRef uintptr,
	inputRef, outRef uintptr,
	format dtype.DType,
	rank, count uint32,
	permutation, inputStrides, outputStrides []uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || len(permutation) != int(rank) ||
		len(inputStrides) != int(rank) || len(outputStrides) != int(rank) {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_transpose(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(inputRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(rank),
		C.uint32_t(count),
		(*C.uint32_t)(unsafe.Pointer(&permutation[0])),
		(*C.uint32_t)(unsafe.Pointer(&inputStrides[0])),
		(*C.uint32_t)(unsafe.Pointer(&outputStrides[0])),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}
