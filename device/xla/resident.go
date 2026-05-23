//go:build xla

package xla

/*
#cgo CXXFLAGS: -I${SRCDIR}/internal/bridge -std=c++17
#include "internal/bridge/core.h"
*/
import "C"

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type residentTensor interface {
	bufferRef() C.XLABufferRef
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

func resolveBufferRef(ptr unsafe.Pointer) C.XLABufferRef {
	deviceTensor := resolveDeviceTensor(ptr)

	if deviceTensor != nil {
		return deviceTensor.bufferRef()
	}

	return C.XLABufferRef(ptr)
}

func residentPointer(deviceTensor *DeviceTensor) unsafe.Pointer {
	return unsafe.Pointer(deviceTensor)
}
