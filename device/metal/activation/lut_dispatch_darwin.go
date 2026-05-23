//go:build darwin && cgo

package activation

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
)

/*
#include <string.h>
#include "lut.h"
#include "../internal/bridge/core.h"
*/
import "C"

var lutBufferCache sync.Map

func standardKernelOperationName(kernel StandardKernel) string {
	switch kernel {
	case StandardExp:
		return "exp"
	case StandardLog:
		return "log"
	case StandardLog1p:
		return "log1p"
	case StandardExpm1:
		return "expm1"
	case StandardSigmoid:
		return "sigmoid"
	case StandardLogSigmoid:
		return "log_sigmoid"
	case StandardTanh:
		return "tanh"
	case StandardSilu:
		return "silu"
	case StandardSwish:
		return "swish"
	case StandardGeluTanh:
		return "gelu_tanh"
	case StandardGelu:
		return "gelu"
	case StandardReLU:
		return "relu"
	case StandardLeakyReLU:
		return "leaky_relu"
	case StandardELU:
		return "elu"
	case StandardCELU:
		return "celu"
	case StandardSELU:
		return "selu"
	case StandardSoftplus:
		return "softplus"
	case StandardMish:
		return "mish"
	case StandardSoftsign:
		return "softsign"
	case StandardHardSigmoid:
		return "hardsigmoid"
	case StandardHardSwish:
		return "hardswish"
	case StandardHardTanh:
		return "hardtanh"
	case StandardHardGelu:
		return "hard_gelu"
	case StandardQuickGelu:
		return "quick_gelu"
	case StandardTanhShrink:
		return "tanh_shrink"
	default:
		return ""
	}
}

func productionLUTTable(kernel StandardKernel, format dtype.DType) (*[65536]uint16, bool) {
	operationName := standardKernelOperationName(kernel)

	if operationName == "" {
		return nil, false
	}

	return cpuactivation.LUTTable(operationName, format)
}

func lutBufferRef(contextRef C.MetalDeviceRef, lutTable *[65536]uint16) C.MetalBufferRef {
	if lutTable == nil {
		return nil
	}

	cacheKey := uintptr(unsafe.Pointer(lutTable))

	if cached, ok := lutBufferCache.Load(cacheKey); ok {
		return cached.(C.MetalBufferRef)
	}

	buffer := C.metal_buffer_new_shared(contextRef, C.longlong(len(lutTable)*2))

	if buffer == nil {
		return nil
	}

	contents := C.metal_buffer_contents(buffer)

	if contents == nil {
		C.metal_buffer_release(buffer)
		return nil
	}

	C.memcpy(contents, unsafe.Pointer(&lutTable[0]), C.size_t(len(lutTable)*2))
	lutBufferCache.Store(cacheKey, buffer)

	return buffer
}

/*
DispatchLUTGather applies a production LUT table to f16/bf16 storage lanes.
*/
func DispatchLUTGather(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	srcBuffer C.MetalBufferRef,
	format dtype.DType,
	lutTable *[65536]uint16,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	lutRef := lutBufferRef(contextRef, lutTable)

	if lutRef == nil {
		return errNilLUTBuffer
	}

	var status C.MetalStatus
	code := C.metal_dispatch_lut_gather(
		contextRef,
		elementFormat,
		srcBuffer,
		dstBuffer,
		lutRef,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

var errNilLUTBuffer = errors.New("metal activation: nil LUT buffer")
