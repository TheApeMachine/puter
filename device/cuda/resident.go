//go:build cuda

package cuda

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type residentTensor interface {
	bufferRef() C.CUDABufferRef
	format() dtype.DType
	storageBytes() int64
}

func resolveDeviceTensor(ptr unsafe.Pointer) *DeviceTensor {
	if ptr == nil {
		return nil
	}

	deviceTensor := (*DeviceTensor)(ptr)

	if deviceTensor.buffer == nil {
		return nil
	}

	return deviceTensor
}

func resolveBufferRef(ptr unsafe.Pointer) C.CUDABufferRef {
	deviceTensor := resolveDeviceTensor(ptr)

	if deviceTensor != nil {
		return deviceTensor.bufferRef()
	}

	return C.CUDABufferRef(ptr)
}

func unsafeBytes(bytesIn []byte) unsafe.Pointer {
	if len(bytesIn) == 0 {
		return nil
	}

	return unsafe.Pointer(&bytesIn[0])
}

func bridgeStatusError(status C.CUDAStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensorErrFromStatus(status)
}
