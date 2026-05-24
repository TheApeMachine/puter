//go:build darwin && cgo

package elementwise

import (
	"errors"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "arithmetic.h"
#include "math.h"
#include "axpy.h"
*/
import "C"

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

func unaryMathOperation(kernel UnaryKernel) C.int {
	switch kernel {
	case UnaryAbs:
		return C.MetalUnaryMathAbs
	case UnaryNeg:
		return C.MetalUnaryMathNeg
	case UnarySqrt:
		return C.MetalUnaryMathSqrt
	case UnaryReLU:
		return C.MetalUnaryMathReLU
	default:
		return -1
	}
}

/*
binaryArithmeticOperation maps BinaryKernel to Metal operation codes.
MetalBinaryFloat32* names are historical: elementFormat selects storage precision.
*/
func binaryArithmeticOperation(kernel BinaryKernel) C.int {
	switch kernel {
	case BinaryAdd:
		return C.MetalBinaryFloat32Add
	case BinarySub:
		return C.MetalBinaryFloat32Sub
	case BinaryMul:
		return C.MetalBinaryFloat32Mul
	case BinaryDiv:
		return C.MetalBinaryFloat32Div
	case BinaryMax:
		return C.MetalBinaryFloat32Max
	case BinaryMin:
		return C.MetalBinaryFloat32Min
	default:
		return -1
	}
}

func DispatchUnaryMath(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	srcBuffer C.MetalBufferRef,
	format dtype.DType,
	kernel UnaryKernel,
	count uint32,
) error {
	operation := unaryMathOperation(kernel)

	if operation < 0 {
		return errUnsupportedKernel
	}

	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_unary_math(
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

func DispatchBinaryElementwise(
	contextRef C.MetalDeviceRef,
	dstBuffer C.MetalBufferRef,
	leftBuffer C.MetalBufferRef,
	rightBuffer C.MetalBufferRef,
	format dtype.DType,
	kernel BinaryKernel,
	count uint32,
) error {
	operation := binaryArithmeticOperation(kernel)

	if operation < 0 {
		return errUnsupportedKernel
	}

	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_binary_elementwise(
		contextRef,
		operation,
		elementFormat,
		leftBuffer,
		rightBuffer,
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

func DispatchAxpy(
	contextRef C.MetalDeviceRef,
	yBuffer C.MetalBufferRef,
	xBuffer C.MetalBufferRef,
	format dtype.DType,
	alpha float32,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.MetalStatus
	code := C.metal_dispatch_axpy(
		contextRef,
		elementFormat,
		yBuffer,
		xBuffer,
		C.float(alpha),
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchBinaryElementwiseRefs(
	contextRef unsafe.Pointer,
	dstBuffer unsafe.Pointer,
	leftBuffer unsafe.Pointer,
	rightBuffer unsafe.Pointer,
	format dtype.DType,
	kernel BinaryKernel,
	count uint32,
) error {
	return DispatchBinaryElementwise(
		C.MetalDeviceRef(contextRef),
		C.MetalBufferRef(dstBuffer),
		C.MetalBufferRef(leftBuffer),
		C.MetalBufferRef(rightBuffer),
		format,
		kernel,
		count,
	)
}

func DispatchUnaryMathRefs(
	contextRef unsafe.Pointer,
	dstBuffer unsafe.Pointer,
	srcBuffer unsafe.Pointer,
	format dtype.DType,
	kernel UnaryKernel,
	count uint32,
) error {
	return DispatchUnaryMath(
		C.MetalDeviceRef(contextRef),
		C.MetalBufferRef(dstBuffer),
		C.MetalBufferRef(srcBuffer),
		format,
		kernel,
		count,
	)
}

func DispatchAxpyRefs(
	contextRef unsafe.Pointer,
	yBuffer unsafe.Pointer,
	xBuffer unsafe.Pointer,
	format dtype.DType,
	alpha float32,
	count uint32,
) error {
	return DispatchAxpy(
		C.MetalDeviceRef(contextRef),
		C.MetalBufferRef(yBuffer),
		C.MetalBufferRef(xBuffer),
		format,
		alpha,
		count,
	)
}

var (
	errUnsupportedDType  = errors.New("metal elementwise: unsupported dtype")
	errUnsupportedKernel = errors.New("metal elementwise: unsupported kernel")
)

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
