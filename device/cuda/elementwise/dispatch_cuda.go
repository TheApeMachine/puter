//go:build cuda

package elementwise

import (
	_ "embed"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "elementwise.h"
#include "math.h"
#include "arithmetic.h"
#include "axpy.h"
*/
import "C"

//go:embed elementwise.cuh
var elementwiseHubSource string

//go:embed math.cu
var mathDomainSource string

//go:embed arithmetic.cu
var arithmeticDomainSource string

//go:embed axpy.cu
var axpyDomainSource string

func moduleSource() string {
	parts := []string{
		elementwiseHubSource,
		mathDomainSource,
		arithmeticDomainSource,
		axpyDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_elementwise_register_module_source(source)
}

func init() {
	registerModuleSource()
}

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.CUDAElementDTypeFloat32
	case dtype.Float16:
		return C.CUDAElementDTypeFloat16
	case dtype.BFloat16:
		return C.CUDAElementDTypeBFloat16
	case dtype.Float64:
		return C.CUDAElementDTypeFloat64
	case dtype.Float8E4M3:
		return C.CUDAElementDTypeFloat8E4M3
	case dtype.Float8E5M2:
		return C.CUDAElementDTypeFloat8E5M2
	default:
		return -1
	}
}

func cudaStatusError(status C.CUDAStatus) error {
	if status.code == 0 {
		return nil
	}

	message := C.GoString(&status.message[0])
	return &dispatchError{code: int(status.code), message: message}
}

type dispatchError struct {
	code    int
	message string
}

func (dispatchError *dispatchError) Error() string {
	return dispatchError.message
}

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA elementwise dtype"}

func unaryMathOperation(kernel UnaryKernel) C.int {
	switch kernel {
	case UnaryAbs:
		return C.CUDAUnaryMathAbs
	case UnaryNeg:
		return C.CUDAUnaryMathNeg
	case UnarySqrt:
		return C.CUDAUnaryMathSqrt
	case UnaryReLU:
		return C.CUDAUnaryMathReLU
	default:
		return -1
	}
}

func binaryArithmeticOperation(kernel BinaryKernel) C.int {
	switch kernel {
	case BinaryAdd:
		return C.CUDABinaryFloat32Add
	case BinarySub:
		return C.CUDABinaryFloat32Sub
	case BinaryMul:
		return C.CUDABinaryFloat32Mul
	case BinaryDiv:
		return C.CUDABinaryFloat32Div
	case BinaryMax:
		return C.CUDABinaryFloat32Max
	case BinaryMin:
		return C.CUDABinaryFloat32Min
	default:
		return -1
	}
}

/*
DispatchUnaryMath launches a unary elementwise math kernel.
*/
func DispatchUnaryMath(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	srcBuffer C.CUDABufferRef,
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

	var status C.CUDAStatus
	code := C.cuda_dispatch_unary_math(
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
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchBinaryElementwise launches a binary elementwise kernel.
*/
func DispatchBinaryElementwise(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
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

	var status C.CUDAStatus
	code := C.cuda_dispatch_binary_elementwise(
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
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchAxpy launches y += alpha * x on device.
*/
func DispatchAxpy(
	contextRef C.CUDADeviceRef,
	yBuffer C.CUDABufferRef,
	xBuffer C.CUDABufferRef,
	format dtype.DType,
	alpha float32,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_axpy(
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
		return cudaStatusError(status)
	}

	return nil
}

var errUnsupportedKernel = &dispatchError{code: -6, message: "unsupported CUDA elementwise kernel"}

func DispatchBinaryElementwiseRefs(
	contextRef uintptr,
	dstBuffer uintptr,
	leftBuffer uintptr,
	rightBuffer uintptr,
	format dtype.DType,
	kernel BinaryKernel,
	count uint32,
) error {
	return DispatchBinaryElementwise(
		C.CUDADeviceRef(unsafe.Pointer(contextRef)),
		C.CUDABufferRef(unsafe.Pointer(dstBuffer)),
		C.CUDABufferRef(unsafe.Pointer(leftBuffer)),
		C.CUDABufferRef(unsafe.Pointer(rightBuffer)),
		format,
		kernel,
		count,
	)
}
