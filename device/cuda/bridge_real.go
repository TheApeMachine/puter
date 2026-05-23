//go:build cuda

package cuda

/*
#cgo cuda CFLAGS: -I${SRCDIR}/internal/bridge
#cgo cuda LDFLAGS: -lcuda -lcudart -lnvrtc -lpthread

#include "internal/bridge/core.h"

extern int cuda_open_device(CUDADeviceRef* outDevice, CUDAStatus* status);
extern void cuda_close_device(CUDADeviceRef device);
extern long long cuda_device_total_memory(CUDADeviceRef device);
extern int cuda_device_capability_major(CUDADeviceRef device);
extern CUDABufferRef cuda_buffer_alloc(CUDADeviceRef device, long long bytes);
extern void cuda_buffer_release(CUDABufferRef buffer);
extern int cuda_memcpy_async_h2d(CUDABufferRef dst, const void* src, long long bytes, CUDAStreamRef stream, CUDAStatus* status);
extern int cuda_memcpy_async_d2h(void* dst, CUDABufferRef src, long long bytes, CUDAStreamRef stream, CUDAStatus* status);
extern int cuda_stream_synchronize(CUDAStreamRef stream, CUDAStatus* status);
extern CUDADeviceRef cuda_default_context(void);
extern CUDAStreamRef cuda_context_upload_stream(CUDADeviceRef device);
*/
import "C"

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type cudaBridge struct {
	device       C.CUDADeviceRef
	uploadStream C.CUDAStreamRef
	backend      *Backend
	dtypes       []dtype.DType
	totalMem     int64
}

func openCUDABridge(backend *Backend) (*cudaBridge, error) {
	var device C.CUDADeviceRef
	var status C.CUDAStatus
	code := C.cuda_open_device(&device, &status)

	if code != 0 || device == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	major := int(C.cuda_device_capability_major(device))
	totalMem := int64(C.cuda_device_total_memory(device))

	supported := []dtype.DType{
		dtype.Float32,
		dtype.BFloat16,
		dtype.Float16,
		dtype.Int8,
		dtype.Int4,
		dtype.Bool,
	}

	if major >= 9 {
		supported = append(supported, dtype.Float8E4M3, dtype.Float8E5M2)
	}

	contextUploadStream := C.cuda_context_upload_stream(device)

	return &cudaBridge{
		device:       device,
		uploadStream: contextUploadStream,
		backend:      backend,
		dtypes:       supported,
		totalMem:     totalMem,
	}, nil
}

func (bridge *cudaBridge) contextRef() C.CUDADeviceRef {
	return bridge.device
}

func (bridge *cudaBridge) supportedDTypes() []dtype.DType {
	return bridge.dtypes
}

func (bridge *cudaBridge) totalGlobalMem() int64 {
	return bridge.totalMem
}

func (bridge *cudaBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	tensorValue, err := bridge.uploadAsync(shape, sourceDType, bytesIn)

	if err != nil {
		return nil, err
	}

	if waitErr := tensorValue.(*DeviceTensor).WaitReady(); waitErr != nil {
		return nil, waitErr
	}

	return tensorValue, nil
}

func (bridge *cudaBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return bridge.stageUpload(shape, sourceDType, bytesIn, true)
}

func (bridge *cudaBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	deviceTensor, ok := input.(*DeviceTensor)

	if !ok {
		return dtype.Invalid, nil, tensor.ErrShapeMismatch
	}

	bytesOut := make([]byte, deviceTensor.byteCount)
	var status C.CUDAStatus
	code := C.cuda_memcpy_async_d2h(
		unsafeBytes(bytesOut),
		deviceTensor.bufferRef(),
		C.longlong(deviceTensor.byteCount),
		bridge.uploadStream,
		&status,
	)

	if code != 0 {
		return dtype.Invalid, nil, bridgeStatusError(status)
	}

	if syncCode := C.cuda_stream_synchronize(bridge.uploadStream, &status); syncCode != 0 {
		return dtype.Invalid, nil, bridgeStatusError(status)
	}

	return deviceTensor.format(), bytesOut, nil
}

func (bridge *cudaBridge) close() error {
	if bridge.device != nil {
		C.cuda_close_device(bridge.device)
		bridge.device = nil
	}

	return nil
}

func (bridge *cudaBridge) stageUpload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
	async bool,
) (tensor.Tensor, error) {
	buffer := C.cuda_buffer_alloc(bridge.device, C.longlong(len(bytesIn)))

	if buffer == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	var status C.CUDAStatus
	code := C.cuda_memcpy_async_h2d(
		buffer,
		unsafeBytes(bytesIn),
		C.longlong(len(bytesIn)),
		bridge.uploadStream,
		&status,
	)

	if code != 0 {
		C.cuda_buffer_release(buffer)
		return nil, bridgeStatusError(status)
	}

	if !async {
		if syncCode := C.cuda_stream_synchronize(bridge.uploadStream, &status); syncCode != 0 {
			C.cuda_buffer_release(buffer)
			return nil, bridgeStatusError(status)
		}
	}

	return newDeviceTensor(
		bridge.backend,
		shape,
		sourceDType,
		buffer,
		len(bytesIn),
		async,
	), nil
}
