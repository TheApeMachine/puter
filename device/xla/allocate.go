//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const workspaceAlign = 128

func (backend *Backend) allocateAligned(byteCount int64, elementFormat dtype.DType, shape tensor.Shape) (unsafe.Pointer, error) {
	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	bytes := make([]byte, byteCount)
	deviceTensor, err := backend.bridge.stageUpload(shape, elementFormat, bytes, false)

	if err != nil {
		return nil, err
	}

	return unsafe.Pointer(deviceTensor.(*DeviceTensor)), nil
}

func (backend *Backend) release(devicePointer unsafe.Pointer) {
	deviceTensor := resolveDeviceTensor(devicePointer)

	if deviceTensor == nil {
		return
	}

	_ = deviceTensor.Close()
}

func alignUp(value int64, alignment int64) int64 {
	if alignment <= 0 {
		return value
	}

	remainder := value % alignment

	if remainder == 0 {
		return value
	}

	return value + alignment - remainder
}
