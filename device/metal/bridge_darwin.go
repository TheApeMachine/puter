//go:build darwin && cgo

package metal

import (
	"context"
	_ "embed"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

/*
#cgo CFLAGS: -I${SRCDIR}/internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "internal/bridge/core.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"

//go:embed kernels.metallib
var kernelsMetalLibrary []byte

type metalBridge struct {
	device  C.MetalDeviceRef
	backend *Backend
}

func openMetalBridge(backend *Backend) (*metalBridge, error) {
	if len(kernelsMetalLibrary) == 0 {
		return nil, fmt.Errorf("%w: empty Metal library", tensor.ErrNeedsPlatformSetup)
	}

	status := C.MetalStatus{}
	device := C.metal_open_default_device(
		(*C.uint8_t)(unsafe.Pointer(&kernelsMetalLibrary[0])),
		C.longlong(len(kernelsMetalLibrary)),
		&status,
	)
	runtime.KeepAlive(kernelsMetalLibrary)

	if device == nil {
		return nil, fmt.Errorf("%w: %s", tensor.ErrNeedsPlatformSetup, bridgeStatusMessage(status))
	}

	return &metalBridge{
		device:  device,
		backend: backend,
	}, nil
}

func (bridge *metalBridge) contextRef() C.MetalDeviceRef {
	return bridge.device
}

func (bridge *metalBridge) recommendedMaxWorkingSet() int64 {
	if bridge == nil || bridge.device == nil {
		return 0
	}

	return int64(C.metal_recommended_max_working_set(bridge.device))
}

func (bridge *metalBridge) close() error {
	if bridge.device != nil {
		C.metal_device_release(bridge.device)
		bridge.device = nil
	}

	return nil
}

func (bridge *metalBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	byteCount, err := shape.Bytes(sourceDType)

	if err != nil {
		return nil, err
	}

	if byteCount != len(bytesIn) {
		return nil, tensor.ErrShapeMismatch
	}

	var buffer C.MetalBufferRef

	if byteCount > 0 {
		buffer = C.metal_buffer_new_shared(bridge.device, C.longlong(byteCount))

		if buffer == nil {
			return nil, tensor.ErrAllocatorExhausted
		}

		contents := C.metal_buffer_contents(buffer)

		if contents == nil {
			C.metal_buffer_release(buffer)
			return nil, tensor.ErrNeedsPlatformSetup
		}

		C.memcpy(contents, unsafe.Pointer(&bytesIn[0]), C.size_t(byteCount))
	}

	deviceTensor := newDeviceTensor(bridge.backend, shape, sourceDType, buffer, byteCount)
	return deviceTensor, nil
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	deviceTensor, err := requireDeviceTensor(input)

	if err != nil {
		return dtype.Invalid, nil, err
	}

	if deviceTensor.byteCount == 0 {
		return deviceTensor.elementFormat, []byte{}, nil
	}

	contents := C.metal_buffer_contents(deviceTensor.buffer)

	if contents == nil {
		return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
	}

	bytesOut := make([]byte, deviceTensor.byteCount)
	C.memcpy(unsafe.Pointer(&bytesOut[0]), contents, C.size_t(deviceTensor.byteCount))

	return deviceTensor.elementFormat, bytesOut, nil
}

func bridgeStatusMessage(status C.MetalStatus) string {
	if status.code == 0 {
		return "metal bridge ok"
	}

	return C.GoString(&status.message[0])
}

/*
NewBackend constructs a Metal backend on darwin+cgo builds.
*/
func NewBackend(ctx context.Context, workerPool *qpool.Q) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	backend := &Backend{
		ctx:    ctx,
		cancel: cancel,
		pool:   workerPool,
	}

	bridge, err := openMetalBridge(backend)

	if err != nil {
		cancel()
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge}
	backend.bindFamilies(computeHost)

	return backend, nil
}
