//go:build darwin && cgo

package metal

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}
#include "internal/bridge/core.h"

extern int metal_dispatch_activation_steer_float32(
	MetalDeviceRef contextRef,
	MetalBufferRef destinationRef,
	MetalBufferRef baseRef,
	MetalBufferRef directionRef,
	MetalBufferRef coefficientRef,
	uint32_t count,
	uint64_t completionToken,
	MetalStatus* status
);
*/
import "C"

func (host *ComputeHost) DispatchActivationSteer(
	destination, base, direction unsafe.Pointer,
	coefficient float32,
	count int,
	format dtype.DType,
) {
	if format != dtype.Float32 {
		host.dispatchError(fmt.Errorf("interpretability: unsupported dtype %v", format))
	}

	if count == 0 {
		return
	}

	contextRef := C.MetalDeviceRef(unsafe.Pointer(host.devicePointer()))

	coefficientBuffer := C.metal_buffer_new_shared(contextRef, 4)

	if coefficientBuffer == nil {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	defer C.metal_buffer_release(coefficientBuffer)

	coefficientContents := C.metal_buffer_contents(coefficientBuffer)
	*(*C.float)(coefficientContents) = C.float(coefficient)

	status := C.MetalStatus{}
	code := C.metal_dispatch_activation_steer_float32(
		contextRef,
		C.MetalBufferRef(unsafe.Pointer(resolveBufferRef(destination))),
		C.MetalBufferRef(unsafe.Pointer(resolveBufferRef(base))),
		C.MetalBufferRef(unsafe.Pointer(resolveBufferRef(direction))),
		coefficientBuffer,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		host.dispatchError(activationSteerStatusError(status))
	}
}

func activationSteerStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	message := C.GoString(&status.message[0])

	return &activationSteerDispatchError{message: message}
}

type activationSteerDispatchError struct {
	message string
}

func (dispatchError *activationSteerDispatchError) Error() string {
	return dispatchError.message
}
