//go:build cuda

package matmul

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "matmul.h"
#include "product.h"
*/
import "C"

//go:embed matmul.cuh
var matmulHubSource string

//go:embed product.cu
var productDomainSource string

func moduleSource() string {
	parts := []string{
		matmulHubSource,
		productDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_matmul_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA matmul dtype"}

/*
DispatchMatmul launches a tiled CUDA GEMM kernel.
*/
func DispatchMatmul(
	contextRef C.CUDADeviceRef,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
	format dtype.DType,
	rows uint32,
	inner uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_matmul(
		contextRef,
		elementFormat,
		leftBuffer,
		rightBuffer,
		outBuffer,
		C.uint32_t(rows),
		C.uint32_t(inner),
		C.uint32_t(cols),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
