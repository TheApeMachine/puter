//go:build darwin && cgo

package metal

import "unsafe"

func resolveDeviceTensor(pointer unsafe.Pointer) *DeviceTensor {
	if pointer == nil {
		return nil
	}

	deviceTensor := (*DeviceTensor)(pointer)

	if deviceTensor.buffer == nil {
		return nil
	}

	return deviceTensor
}

func resolveBufferRef(pointer unsafe.Pointer) C.MetalBufferRef {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor != nil {
		return deviceTensor.bufferRef()
	}

	return C.MetalBufferRef(pointer)
}
