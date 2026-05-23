//go:build cuda

package cuda

/*
#include "internal/bridge/core.h"

extern CUDAStreamRef cuda_context_default_stream(CUDADeviceRef device);
*/
import "C"

import (
	"math"
	"unsafe"
)

func (bridge *cudaBridge) borrowScratch(byteCount int64) C.CUDABufferRef {
	if byteCount <= 0 {
		return nil
	}

	bufferRef := C.cuda_buffer_alloc(bridge.device, C.longlong(byteCount))

	if bufferRef == nil {
		panic("cuda: scratch allocation failed")
	}

	return bufferRef
}

func (bridge *cudaBridge) releaseScratch(bufferRef C.CUDABufferRef) {
	if bufferRef == nil {
		return
	}

	C.cuda_buffer_release(bufferRef)
}

func (bridge *cudaBridge) readFloat32Scalar(bufferRef C.CUDABufferRef) float32 {
	if bufferRef == nil {
		return 0
	}

	var value float32
	var status C.CUDAStatus
	stream := C.cuda_context_default_stream(bridge.device)
	code := C.cuda_memcpy_async_d2h(
		unsafe.Pointer(&value),
		bufferRef,
		C.longlong(4),
		stream,
		&status,
	)

	if code != 0 {
		panic(bridgeStatusError(status))
	}

	if syncCode := C.cuda_stream_synchronize(stream, &status); syncCode != 0 {
		panic(bridgeStatusError(status))
	}

	return value
}

func (bridge *cudaBridge) readInt32Scalar(bufferRef C.CUDABufferRef) int32 {
	if bufferRef == nil {
		return 0
	}

	var value int32
	var status C.CUDAStatus
	stream := C.cuda_context_default_stream(bridge.device)
	code := C.cuda_memcpy_async_d2h(
		unsafe.Pointer(&value),
		bufferRef,
		C.longlong(4),
		stream,
		&status,
	)

	if code != 0 {
		panic(bridgeStatusError(status))
	}

	if syncCode := C.cuda_stream_synchronize(stream, &status); syncCode != 0 {
		panic(bridgeStatusError(status))
	}

	return value
}

func partialReductionCount(count uint32) uint32 {
	return (count + 255) / 256
}

func dotScratchBytes(count uint32) int64 {
	partialCount := partialReductionCount(count)
	return int64(partialCount) * 8
}

func reductionScratchBytes(count uint32) int64 {
	partialCount := partialReductionCount(count)
	return int64(partialCount) * 4
}

func crossEntropyScratchBytes(batch uint32) int64 {
	return int64(batch) * 4
}

func samplingScoresScratchBytes(paddedCount uint32) int64 {
	return int64(paddedCount) * 4
}

func samplingIndicesScratchBytes(paddedCount uint32) int64 {
	return int64(paddedCount) * 4
}

func attentionScoresBytes(seqQ, seqK int) int64 {
	if seqQ <= 0 || seqK <= 0 {
		return 0
	}

	return int64(seqQ) * int64(seqK) * 4
}

func causalIvScratchBytes(count uint32) int64 {
	partialCount := partialReductionCount(count)
	return int64(partialCount) * 5 * 4
}

func causalDagScratchBytes(count uint32) int64 {
	partialCount := partialReductionCount(count)
	return int64(partialCount) * 4
}

func causalScalarBytes(format dtype.DType) int64 {
	switch format {
	case dtype.Float32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 0
	}
}

func (bridge *cudaBridge) writeDeviceScalar(bufferRef C.CUDABufferRef, value float32, format dtype.DType) {
	if bufferRef == nil {
		panic("cuda: nil scalar buffer")
	}

	var payload [4]byte
	var byteCount int64

	switch format {
	case dtype.Float32:
		math.Float32bits(value)
		*(*float32)(unsafe.Pointer(&payload[0])) = value
		byteCount = 4
	case dtype.Float16:
		encoded := dtype.Fromfloat32(value)
		payload[0] = byte(encoded)
		payload[1] = byte(encoded >> 8)
		byteCount = 2
	case dtype.BFloat16:
		encoded := dtype.NewBfloat16FromFloat32(value)
		payload[0] = byte(encoded)
		payload[1] = byte(encoded >> 8)
		byteCount = 2
	default:
		panic("cuda: unsupported scalar dtype")
	}

	var status C.CUDAStatus
	code := C.cuda_memcpy_async_h2d(
		bufferRef,
		unsafe.Pointer(&payload[0]),
		C.longlong(byteCount),
		bridge.uploadStream,
		&status,
	)

	if code != 0 {
		panic(bridgeStatusError(status))
	}

	if syncCode := C.cuda_stream_synchronize(bridge.uploadStream, &status); syncCode != 0 {
		panic(bridgeStatusError(status))
	}
}

func ropePairCount(seqLen, numHeads, headDim int) uint32 {
	halfDim := headDim / 2

	if seqLen <= 0 || numHeads <= 0 || halfDim <= 0 {
		return 0
	}

	return uint32(seqLen * numHeads * halfDim)
}

func conv2dLaunchCount(batch, outChannels, outHeight, outWidth int) uint32 {
	if batch <= 0 || outChannels <= 0 || outHeight <= 0 || outWidth <= 0 {
		return 0
	}

	total := int64(batch) * int64(outChannels) * int64(outHeight) * int64(outWidth)

	if total > math.MaxUint32 {
		panic("cuda: conv2d launch count overflow")
	}

	return uint32(total)
}
