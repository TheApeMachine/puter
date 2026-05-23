//go:build cuda

package cuda

/*
#include "internal/bridge/core.h"

extern int cuda_memcpy_async_h2d(CUDABufferRef dst, const void* src, long long bytes, CUDAStreamRef stream, CUDAStatus* status);
extern int cuda_stream_synchronize(CUDAStreamRef stream, CUDAStatus* status);
extern CUDAStreamRef cuda_context_default_stream(CUDADeviceRef device);
*/
import "C"

import (
	"github.com/theapemachine/manifesto/dtype"
)

func physicsSpacingBytes(spacing float32, format dtype.DType) []byte {
	switch format {
	case dtype.Float32:
		return dtype.Float32ToBytes([]float32{spacing})
	case dtype.Float16:
		value := uint16(dtype.Fromfloat32(spacing))
		return []byte{byte(value), byte(value >> 8)}
	case dtype.BFloat16:
		value := dtype.NewBfloat16FromFloat32(spacing)
		return value.Encode([]dtype.BF16{value})
	default:
		return nil
	}
}

func (bridge *cudaBridge) uploadHostBytes(bytes []byte) C.CUDABufferRef {
	if len(bytes) == 0 {
		return nil
	}

	bufferRef := bridge.borrowScratch(int64(len(bytes)))
	var status C.CUDAStatus
	stream := C.cuda_context_default_stream(bridge.device)
	code := C.cuda_memcpy_async_h2d(
		bufferRef,
		unsafeBytes(bytes),
		C.longlong(len(bytes)),
		stream,
		&status,
	)

	if code != 0 {
		bridge.releaseScratch(bufferRef)
		panic(bridgeStatusError(status))
	}

	if syncCode := C.cuda_stream_synchronize(stream, &status); syncCode != 0 {
		bridge.releaseScratch(bufferRef)
		panic(bridgeStatusError(status))
	}

	return bufferRef
}
