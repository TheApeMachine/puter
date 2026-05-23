//go:build cuda

package cuda

/*
#include "internal/bridge/core.h"

#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

/*
pipeline resolves a JIT-compiled NVRTC kernel from module source.
The module cache lives in the device context (internal/bridge/context.c).
*/
func (backend *Backend) pipeline(moduleSource, kernelName string) (uintptr, error) {
	if backend.bridge == nil {
		return 0, tensor.ErrNeedsPlatformSetup
	}

	return backend.bridge.pipeline(moduleSource, kernelName)
}

func (bridge *cudaBridge) pipeline(moduleSource, kernelName string) (uintptr, error) {
	moduleText := C.CString(moduleSource)
	kernelText := C.CString(kernelName)
	defer C.free(unsafe.Pointer(moduleText))
	defer C.free(unsafe.Pointer(kernelText))

	var status C.CUDAStatus
	kernelRef := C.cuda_bridge_resolve_kernel(
		bridge.device,
		moduleText,
		kernelText,
		&status,
	)

	if kernelRef == nil {
		return 0, bridgeStatusError(status)
	}

	return uintptr(unsafe.Pointer(kernelRef)), nil
}
