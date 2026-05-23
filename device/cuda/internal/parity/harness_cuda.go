//go:build cuda

package parity

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo cuda CFLAGS: -I${SRCDIR}/../bridge
#cgo cuda LDFLAGS: -lcuda -lcudart -lpthread

#include "../bridge/core.h"

extern int cuda_open_device(CUDADeviceRef* outDevice, CUDAStatus* status);
extern void cuda_close_device(CUDADeviceRef device);
extern CUDABufferRef cuda_buffer_alloc(CUDADeviceRef device, long long bytes);
extern void cuda_buffer_release(CUDABufferRef buffer);
extern int cuda_memcpy_async_h2d(CUDABufferRef dst, const void* src, long long bytes, CUDAStreamRef stream, CUDAStatus* status);
extern int cuda_memcpy_async_d2h(void* dst, CUDABufferRef src, long long bytes, CUDAStreamRef stream, CUDAStatus* status);
extern int cuda_stream_synchronize(CUDAStreamRef stream, CUDAStatus* status);
extern CUDAStreamRef cuda_context_upload_stream(CUDADeviceRef device);
*/
import "C"

/*
Harness runs CUDA kernels against CPU scalar references through the real bridge.
*/
type Harness struct {
	device       C.CUDADeviceRef
	uploadStream C.CUDAStreamRef
}

/*
NewHarness opens the CUDA bridge or skips the test when unavailable.
*/
func NewHarness(testingTB testing.TB) *Harness {
	testingTB.Helper()

	var device C.CUDADeviceRef
	var status C.CUDAStatus
	code := C.cuda_open_device(&device, &status)

	if code != 0 || device == nil {
		testingTB.Skipf("CUDA backend unavailable: %s", C.GoString(&status.message[0]))
	}

	uploadStream := C.cuda_context_upload_stream(device)

	if uploadStream == nil {
		C.cuda_close_device(device)
		testingTB.Skip("CUDA upload stream unavailable")
	}

	return &Harness{
		device:       device,
		uploadStream: uploadStream,
	}
}

/*
Close releases the CUDA device.
*/
func (harness *Harness) Close() {
	if harness == nil || harness.device == nil {
		return
	}

	C.cuda_close_device(harness.device)
	harness.device = nil
	harness.uploadStream = nil
}

/*
Sync waits for in-flight CUDA memcpy and kernel work to finish.
*/
func (harness *Harness) Sync() {
	if harness == nil || harness.uploadStream == nil {
		return
	}

	var status C.CUDAStatus
	C.cuda_stream_synchronize(harness.uploadStream, &status)
}

/*
UploadVector copies host values into a device buffer.
*/
func (harness *Harness) UploadVector(values []float32, format dtype.DType) *Buffer {
	bytesIn, err := encodeVector(values, format)

	if err != nil {
		panic(err)
	}

	return harness.uploadBytes(bytesIn)
}

/*
UploadBytes copies raw host bytes into a device buffer.
*/
func (harness *Harness) UploadBytes(bytesIn []byte) *Buffer {
	return harness.uploadBytes(bytesIn)
}

func (harness *Harness) uploadBytes(bytesIn []byte) *Buffer {
	if len(bytesIn) == 0 {
		return &Buffer{
			device:       harness.device,
			uploadStream: harness.uploadStream,
		}
	}

	buffer := C.cuda_buffer_alloc(harness.device, C.longlong(len(bytesIn)))

	if buffer == nil {
		panic(tensor.ErrAllocatorExhausted)
	}

	var status C.CUDAStatus
	code := C.cuda_memcpy_async_h2d(
		buffer,
		unsafe.Pointer(&bytesIn[0]),
		C.longlong(len(bytesIn)),
		harness.uploadStream,
		&status,
	)

	if code != 0 {
		C.cuda_buffer_release(buffer)
		panic(fmt.Errorf("cuda parity upload: %s", C.GoString(&status.message[0])))
	}

	runtime.KeepAlive(bytesIn)

	return &Buffer{
		device:       harness.device,
		buffer:       buffer,
		byteCount:    len(bytesIn),
		uploadStream: harness.uploadStream,
	}
}

/*
DownloadFloat32 reads a device buffer back as float32 lanes for ULP comparison.
*/
func (harness *Harness) DownloadFloat32(buffer *Buffer, format dtype.DType) []float32 {
	harness.Sync()

	bytesOut := buffer.readBytes()
	decoded, err := convert.BytesToFloat32(format, bytesOut)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
Buffer is device storage used by parity tests.
*/
type Buffer struct {
	device       C.CUDADeviceRef
	buffer       C.CUDABufferRef
	byteCount    int
	uploadStream C.CUDAStreamRef
}

func (buffer *Buffer) ref() uintptr {
	return uintptr(unsafe.Pointer(buffer.buffer))
}

/*
Ref exposes the CUDA buffer handle for dispatch calls.
*/
func (buffer *Buffer) Ref() uintptr {
	return buffer.ref()
}

/*
ReadBytes downloads the raw buffer contents after sync.
*/
func (buffer *Buffer) ReadBytes() []byte {
	return buffer.readBytes()
}

func (buffer *Buffer) readBytes() []byte {
	if buffer.byteCount == 0 {
		return []byte{}
	}

	bytesOut := make([]byte, buffer.byteCount)
	var status C.CUDAStatus
	code := C.cuda_memcpy_async_d2h(
		unsafe.Pointer(&bytesOut[0]),
		buffer.buffer,
		C.longlong(buffer.byteCount),
		buffer.uploadStream,
		&status,
	)

	if code != 0 {
		panic(fmt.Errorf("cuda parity download: %s", C.GoString(&status.message[0])))
	}

	if syncCode := C.cuda_stream_synchronize(buffer.uploadStream, &status); syncCode != 0 {
		panic(fmt.Errorf("cuda parity sync: %s", C.GoString(&status.message[0])))
	}

	return bytesOut
}

/*
Close releases the device buffer.
*/
func (buffer *Buffer) Close() {
	if buffer == nil || buffer.buffer == nil {
		return
	}

	C.cuda_buffer_release(buffer.buffer)
	buffer.buffer = nil
}

func (harness *Harness) contextRef() uintptr {
	return uintptr(unsafe.Pointer(harness.device))
}

/*
ContextRef exposes the CUDA device handle for dispatch calls.
*/
func (harness *Harness) ContextRef() uintptr {
	return harness.contextRef()
}
