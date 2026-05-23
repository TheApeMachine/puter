//go:build darwin && cgo

package parity

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}/../bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "core.h"
#include "../bridge/core_darwin.m"
#include <stdlib.h>
#include <string.h>
*/
import "C"

/*
Harness runs Metal kernels against CPU scalar references through the real bridge.
*/
type Harness struct {
	device C.MetalDeviceRef
}

/*
NewHarness opens the Metal bridge or skips the test when unavailable.
*/
func NewHarness(testingTB testing.TB) *Harness {
	testingTB.Helper()

	kernelsMetalLibrary, err := loadKernelsMetalLibrary()

	if err != nil || len(kernelsMetalLibrary) == 0 {
		testingTB.Skip("Metal library unavailable")
	}

	status := C.MetalStatus{}
	deviceRef := C.metal_open_default_device(
		(*C.uint8_t)(unsafe.Pointer(&kernelsMetalLibrary[0])),
		C.longlong(len(kernelsMetalLibrary)),
		&status,
	)
	runtime.KeepAlive(kernelsMetalLibrary)

	if deviceRef == nil {
		testingTB.Skipf("Metal backend unavailable: %s", C.GoString(&status.message[0]))
	}

	return &Harness{device: deviceRef}
}

/*
Close releases the Metal device.
*/
func (harness *Harness) Close() {
	if harness == nil || harness.device == nil {
		return
	}

	C.metal_device_release(harness.device)
	harness.device = nil
}

/*
Sync waits for in-flight Metal command buffers to finish.
*/
func (harness *Harness) Sync() {
	if harness == nil || harness.device == nil {
		return
	}

	C.metal_device_wait_idle(harness.device)
}

/*
UploadVector copies host values into a shared Metal buffer.
*/
func (harness *Harness) UploadVector(values []float32, format dtype.DType) *Buffer {
	bytesIn, err := encodeVector(values, format)

	if err != nil {
		panic(err)
	}

	return harness.uploadBytes(bytesIn)
}

func (harness *Harness) uploadBytes(bytesIn []byte) *Buffer {
	if len(bytesIn) == 0 {
		return &Buffer{device: harness.device}
	}

	buffer := C.metal_buffer_new_shared(harness.device, C.longlong(len(bytesIn)))

	if buffer == nil {
		panic(tensor.ErrAllocatorExhausted)
	}

	contents := C.metal_buffer_contents(buffer)

	if contents == nil {
		C.metal_buffer_release(buffer)
		panic(tensor.ErrNeedsPlatformSetup)
	}

	C.memcpy(contents, unsafe.Pointer(&bytesIn[0]), C.size_t(len(bytesIn)))

	return &Buffer{
		device:    harness.device,
		buffer:    buffer,
		byteCount: len(bytesIn),
	}
}

/*
DownloadFloat32 reads a Metal buffer back as float32 lanes for ULP comparison.
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
Buffer is shared Metal storage used by parity tests.
*/
type Buffer struct {
	device    C.MetalDeviceRef
	buffer    C.MetalBufferRef
	byteCount int
}

func (buffer *Buffer) ref() uintptr {
	return uintptr(unsafe.Pointer(buffer.buffer))
}

/*
Ref exposes the Metal buffer handle for dispatch calls.
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

	contents := C.metal_buffer_contents(buffer.buffer)

	if contents == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}

	bytesOut := make([]byte, buffer.byteCount)
	C.memcpy(unsafe.Pointer(&bytesOut[0]), contents, C.size_t(buffer.byteCount))

	return bytesOut
}

/*
Close releases the Metal buffer.
*/
func (buffer *Buffer) Close() {
	if buffer == nil || buffer.buffer == nil {
		return
	}

	C.metal_buffer_release(buffer.buffer)
	buffer.buffer = nil
}

func (harness *Harness) contextRef() uintptr {
	return uintptr(unsafe.Pointer(harness.device))
}

/*
ContextRef exposes the Metal device handle for dispatch calls.
*/
func (harness *Harness) ContextRef() uintptr {
	return harness.contextRef()
}

/*
RandomUnaryInput fills a deterministic float32 vector for unary parity tests.
*/
func RandomUnaryInput(count int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = rng.Float32()*4.0 - 2.0
	}

	return values
}

func encodeVector(values []float32, format dtype.DType) ([]byte, error) {
	switch format {
	case dtype.Float32:
		return convert.Float32ToBytes(values), nil
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for index, value := range values {
			encoded[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded), nil
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded), nil
	default:
		return nil, fmt.Errorf("metal parity: unsupported dtype %v", format)
	}
}
