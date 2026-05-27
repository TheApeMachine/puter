//go:build darwin && cgo

package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR}/internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "internal/bridge/core.h"
*/
import "C"

func (backend *Backend) AllocateWorkspaceSlot(byteCount int) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	if byteCount <= 0 {
		return nil, tensor.ErrShapeInvalid
	}

	buffer := C.metal_buffer_new_shared(backend.bridge.device, C.longlong(byteCount))

	if buffer == nil {
		return nil, tensor.ErrAllocatorExhausted
	}

	shape, err := tensor.NewShape([]int{byteCount})

	if err != nil {
		C.metal_buffer_release(buffer)

		return nil, err
	}

	return newDeviceTensor(backend, shape, dtype.Int8, buffer, byteCount), nil
}

func (backend *Backend) ViewWorkspaceSlot(
	slot tensor.Tensor,
	shape tensor.Shape,
	elementFormat dtype.DType,
	byteCount int,
) (tensor.Tensor, error) {
	deviceTensor, err := requireDeviceTensor(slot)

	if err != nil {
		return nil, err
	}

	if byteCount < 0 || byteCount > deviceTensor.byteCount {
		return nil, tensor.ErrShapeMismatch
	}

	view := &DeviceTensor{
		backend:       backend,
		shape:         shape,
		elementFormat: elementFormat,
		buffer:        deviceTensor.buffer,
		byteCount:     byteCount,
		ownsBuffer:    false,
		workspaceView: true,
	}

	view.state.Store(uint32(tensor.StateReady))

	return view, nil
}
