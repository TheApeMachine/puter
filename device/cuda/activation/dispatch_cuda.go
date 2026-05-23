//go:build cuda

package activation

import (
	_ "embed"
	"errors"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include <stdlib.h>
#include "standard.h"

extern void cuda_activation_register_module_source(const char* source);
*/
import "C"

//go:embed activation.cuh
var activationHubSource string

//go:embed standard.cu
var standardDomainSource string

func moduleSource() string {
	parts := []string{activationHubSource, standardDomainSource}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_activation_register_module_source(source)
}

func init() {
	registerModuleSource()
}

func standardKernelOperation(kernel StandardKernel) C.int {
	switch kernel {
	case StandardExp:
		return C.CUDAUnaryFloat32Exp
	case StandardLog:
		return C.CUDAUnaryFloat32Log
	case StandardLog1p:
		return C.CUDAUnaryFloat32Log1p
	case StandardExpm1:
		return C.CUDAUnaryFloat32Expm1
	case StandardSigmoid:
		return C.CUDAUnaryFloat32Sigmoid
	case StandardLogSigmoid:
		return C.CUDAUnaryFloat32LogSigmoid
	case StandardTanh:
		return C.CUDAUnaryFloat32Tanh
	case StandardSilu:
		return C.CUDAUnaryFloat32Silu
	case StandardSwish:
		return C.CUDAUnaryFloat32Swish
	case StandardGeluTanh:
		return C.CUDAUnaryFloat32GeluTanh
	case StandardGelu:
		return C.CUDAUnaryFloat32Gelu
	case StandardReLU:
		return C.CUDAUnaryFloat32Relu
	case StandardLeakyReLU:
		return C.CUDAUnaryFloat32LeakyReLU
	case StandardELU:
		return C.CUDAUnaryFloat32ELU
	case StandardCELU:
		return C.CUDAUnaryFloat32CELU
	case StandardSELU:
		return C.CUDAUnaryFloat32SELU
	case StandardSoftplus:
		return C.CUDAUnaryFloat32Softplus
	case StandardMish:
		return C.CUDAUnaryFloat32Mish
	case StandardSoftsign:
		return C.CUDAUnaryFloat32Softsign
	case StandardHardSigmoid:
		return C.CUDAUnaryFloat32HardSigmoid
	case StandardHardSwish:
		return C.CUDAUnaryFloat32HardSwish
	case StandardHardTanh:
		return C.CUDAUnaryFloat32HardTanh
	case StandardHardGelu:
		return C.CUDAUnaryFloat32HardGelu
	case StandardQuickGelu:
		return C.CUDAUnaryFloat32QuickGelu
	case StandardTanhShrink:
		return C.CUDAUnaryFloat32TanhShrink
	default:
		return -1
	}
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

func DispatchStandardUnary(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	srcBuffer C.CUDABufferRef,
	format dtype.DType,
	kernel StandardKernel,
	count uint32,
) error {
	operation := standardKernelOperation(kernel)

	if operation < 0 {
		return errUnsupportedKernel
	}

	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_unary_elementwise(
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

var (
	errUnsupportedKernel = errors.New("cuda activation: unsupported kernel")
	errUnsupportedDType  = errors.New("cuda activation: unsupported dtype")
)

func cudaStatusError(status C.CUDAStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}
