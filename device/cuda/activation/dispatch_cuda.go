//go:build cuda

package activation

import (
	_ "embed"
	"errors"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include <stdlib.h>
#include "standard.h"
#include "parametric.h"
#include "softmax.h"
#include "gated.h"

extern void cuda_activation_register_module_source(const char* source);
*/
import "C"

//go:embed activation.cuh
var activationHubSource string

//go:embed softmax_reduce.cuh
var softmaxReduceSource string

//go:embed standard.cu
var standardDomainSource string

//go:embed parametric.cu
var parametricDomainSource string

//go:embed softmax.cu
var softmaxDomainSource string

//go:embed gated.cu
var gatedDomainSource string

func moduleSource() string {
	parts := []string{
		activationHubSource,
		softmaxReduceSource,
		standardDomainSource,
		parametricDomainSource,
		softmaxDomainSource,
		gatedDomainSource,
	}
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

func DispatchUnaryParam(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	srcBuffer C.CUDABufferRef,
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

	var status C.CUDAStatus
	code := C.cuda_dispatch_unary_param(
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
		return cudaStatusError(status)
	}

	return nil
}

func DispatchSoftmax(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	srcBuffer C.CUDABufferRef,
	format dtype.DType,
	rows uint32,
	cols uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_softmax(
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
		return cudaStatusError(status)
	}

	return nil
}

func DispatchGLUTensors(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	gateBuffer C.CUDABufferRef,
	upBuffer C.CUDABufferRef,
	format dtype.DType,
	variant GLUVariant,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	var code C.int

	switch variant {
	case SwiGLU:
		code = C.cuda_dispatch_swiglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GeGLU:
		code = C.cuda_dispatch_geglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GLU:
		code = C.cuda_dispatch_glu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case ReGLU:
		code = C.cuda_dispatch_reglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case SiGLU:
		code = C.cuda_dispatch_siglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case SeGLU:
		code = C.cuda_dispatch_seglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case LinGLU:
		code = C.cuda_dispatch_linglu(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	case GeGLUTanh:
		code = C.cuda_dispatch_geglu_tanh(
			contextRef, elementFormat, dstBuffer, gateBuffer, upBuffer, C.uint32_t(count), 0, &status,
		)
	default:
		return errUnsupportedKernel
	}

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchGLUPacked(
	contextRef C.CUDADeviceRef,
	dstBuffer C.CUDABufferRef,
	packedBuffer C.CUDABufferRef,
	format dtype.DType,
	variant GLUVariant,
	inner uint32,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	var code C.int

	switch variant {
	case SwiGLU:
		code = C.cuda_dispatch_swiglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GeGLU:
		code = C.cuda_dispatch_geglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GLU:
		code = C.cuda_dispatch_glu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case ReGLU:
		code = C.cuda_dispatch_reglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case SiGLU:
		code = C.cuda_dispatch_siglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case SeGLU:
		code = C.cuda_dispatch_seglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case LinGLU:
		code = C.cuda_dispatch_linglu_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	case GeGLUTanh:
		code = C.cuda_dispatch_geglu_tanh_packed(
			contextRef, elementFormat, dstBuffer, packedBuffer, C.uint32_t(inner), C.uint32_t(count), 0, &status,
		)
	default:
		return errUnsupportedKernel
	}

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
