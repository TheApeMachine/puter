//go:build darwin && cgo

package activation

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <stdlib.h>
#include "standard.h"
#include "parametric.h"
#include "softmax.h"
#include "gated.h"
*/
import "C"

func standardKernelOperation(kernel StandardKernel) C.int {
	switch kernel {
	case StandardExp:
		return C.MetalUnaryFloat32Exp
	case StandardLog:
		return C.MetalUnaryFloat32Log
	case StandardLog1p:
		return C.MetalUnaryFloat32Log1p
	case StandardExpm1:
		return C.MetalUnaryFloat32Expm1
	case StandardSigmoid:
		return C.MetalUnaryFloat32Sigmoid
	case StandardLogSigmoid:
		return C.MetalUnaryFloat32LogSigmoid
	case StandardTanh:
		return C.MetalUnaryFloat32Tanh
	case StandardSilu:
		return C.MetalUnaryFloat32Silu
	case StandardSwish:
		return C.MetalUnaryFloat32Swish
	case StandardGeluTanh:
		return C.MetalUnaryFloat32GeluTanh
	case StandardGelu:
		return C.MetalUnaryFloat32Gelu
	case StandardReLU:
		return C.MetalUnaryFloat32Relu
	case StandardLeakyReLU:
		return C.MetalUnaryFloat32LeakyReLU
	case StandardELU:
		return C.MetalUnaryFloat32ELU
	case StandardCELU:
		return C.MetalUnaryFloat32CELU
	case StandardSELU:
		return C.MetalUnaryFloat32SELU
	case StandardSoftplus:
		return C.MetalUnaryFloat32Softplus
	case StandardMish:
		return C.MetalUnaryFloat32Mish
	case StandardSoftsign:
		return C.MetalUnaryFloat32Softsign
	case StandardHardSigmoid:
		return C.MetalUnaryFloat32HardSigmoid
	case StandardHardSwish:
		return C.MetalUnaryFloat32HardSwish
	case StandardHardTanh:
		return C.MetalUnaryFloat32HardTanh
	case StandardHardGelu:
		return C.MetalUnaryFloat32HardGelu
	case StandardQuickGelu:
		return C.MetalUnaryFloat32QuickGelu
	case StandardTanhShrink:
		return C.MetalUnaryFloat32TanhShrink
	default:
		return -1
	}
}

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.MetalElementDTypeFloat32
	case dtype.Float16:
		return C.MetalElementDTypeFloat16
	case dtype.BFloat16:
		return C.MetalElementDTypeBFloat16
	case dtype.Float64:
		return C.MetalElementDTypeFloat64
	default:
		return -1
	}
}

func DispatchStandardUnary(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	srcBuffer C.MetalBufferRef,
	format dtype.DType,
	kernel StandardKernel,
	count uint32,
) error {
	if format == dtype.Float16 || format == dtype.BFloat16 {
		if lutTable, ok := productionLUTTable(kernel, format); ok {
			return DispatchLUTGather(contextRef, dstBuffer, srcBuffer, format, lutTable, count)
		}
	}

	operation := standardKernelOperation(kernel)

	if operation < 0 {
		return errUnsupportedKernel
	}

	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_unary_elementwise(
		contextRef,
		operation,
		elementFormat,
		srcBuffer,
		dstBuffer,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchUnaryParam(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	srcBuffer C.MetalBufferRef,
	format dtype.DType,
	operationPrefix string,
	param float32,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	operationName := C.CString(operationPrefix)
	defer C.free(unsafe.Pointer(operationName))

	var status C.MetalStatus
	code := C.metal_dispatch_unary_param(
		contextRef,
		operationName,
		elementFormat,
		srcBuffer,
		dstBuffer,
		C.uint32_t(count),
		C.float(param),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchSoftmax(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	srcBuffer C.MetalBufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_softmax(
		contextRef,
		elementFormat,
		srcBuffer,
		dstBuffer,
		C.uint32_t(rows),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchGLUTensors(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	gateBuffer C.MetalBufferRef,
	upBuffer C.MetalBufferRef,
	format dtype.DType,
	variant GLUVariant,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	var code C.int

	switch variant {
	case SwiGLU:
		code = C.metal_dispatch_swiglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GeGLU:
		code = C.metal_dispatch_geglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GLU:
		code = C.metal_dispatch_glu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case ReGLU:
		code = C.metal_dispatch_reglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case SiGLU:
		code = C.metal_dispatch_siglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case SeGLU:
		code = C.metal_dispatch_seglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case LinGLU:
		code = C.metal_dispatch_linglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GeGLUTanh:
		code = C.metal_dispatch_geglu_tanh(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	default:
		return errUnsupportedKernel
	}

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchGLUPacked(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	packedBuffer C.MetalBufferRef,
	format dtype.DType,
	variant GLUVariant,
	inner uint32,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	var code C.int

	switch variant {
	case SwiGLU:
		code = C.metal_dispatch_swiglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GeGLU:
		code = C.metal_dispatch_geglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GLU:
		code = C.metal_dispatch_glu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case ReGLU:
		code = C.metal_dispatch_reglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case SiGLU:
		code = C.metal_dispatch_siglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case SeGLU:
		code = C.metal_dispatch_seglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case LinGLU:
		code = C.metal_dispatch_linglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GeGLUTanh:
		code = C.metal_dispatch_geglu_tanh_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	default:
		return errUnsupportedKernel
	}

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchStandardUnaryRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	srcBuffer uintptr,
	format dtype.DType,
	kernel StandardKernel,
	count uint32,
) error {
	return DispatchStandardUnary(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(dstBuffer)),
		C.MetalBufferRef(unsafe.Pointer(srcBuffer)),
		format,
		kernel,
		count,
	)
}

func DispatchUnaryParamRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	srcBuffer uintptr,
	format dtype.DType,
	operationPrefix string,
	param float32,
	count uint32,
) error {
	return DispatchUnaryParam(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(dstBuffer)),
		C.MetalBufferRef(unsafe.Pointer(srcBuffer)),
		format,
		operationPrefix,
		param,
		count,
	)
}

func DispatchSoftmaxRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	srcBuffer uintptr,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	return DispatchSoftmax(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(dstBuffer)),
		C.MetalBufferRef(unsafe.Pointer(srcBuffer)),
		format,
		rows,
		cols,
	)
}

func DispatchGLUTensorsRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	gateBuffer uintptr,
	upBuffer uintptr,
	format dtype.DType,
	variant GLUVariant,
	count uint32,
) error {
	return DispatchGLUTensors(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(dstBuffer)),
		C.MetalBufferRef(unsafe.Pointer(gateBuffer)),
		C.MetalBufferRef(unsafe.Pointer(upBuffer)),
		format,
		variant,
		count,
	)
}

func DispatchGLUPackedRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	packedBuffer uintptr,
	format dtype.DType,
	variant GLUVariant,
	inner uint32,
	count uint32,
) error {
	return DispatchGLUPacked(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.MetalBufferRef(unsafe.Pointer(dstBuffer)),
		C.MetalBufferRef(unsafe.Pointer(packedBuffer)),
		format,
		variant,
		inner,
		count,
	)
}

var (
	errUnsupportedKernel = errors.New("metal activation: unsupported kernel")
	errUnsupportedDType  = errors.New("metal activation: unsupported dtype")
)

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
