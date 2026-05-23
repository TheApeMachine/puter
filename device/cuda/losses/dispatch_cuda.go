//go:build cuda

package losses

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "losses_dispatch.h"
*/
import "C"

//go:embed losses.cuh
var lossesHubSource string

//go:embed losses.cu
var lossesDomainSource string

func moduleSource() string {
	parts := []string{
		lossesHubSource,
		lossesDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_losses_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA losses dtype"}

var errUnsupportedKernel = &dispatchError{code: -6, message: "unsupported CUDA loss kernel"}

func pairLossOperation(kernel LossKernel) C.int {
	switch kernel {
	case KernelMSE, KernelMAE, KernelHuber, KernelBinaryCrossEntropy, KernelKLDivergence:
		return C.int(kernel)
	default:
		return -1
	}
}

func partialCount(count uint32) uint32 {
	return (count + 255) / 256
}

/*
DispatchPairLoss launches partial + finalize pair-loss kernels on device.
*/
func DispatchPairLoss(
	contextRef C.CUDADeviceRef,
	predictionsBuffer C.CUDABufferRef,
	targetsBuffer C.CUDABufferRef,
	scratchBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
	format dtype.DType,
	kernel LossKernel,
	count uint32,
) error {
	elementFormat := elementDType(format)
	operation := pairLossOperation(kernel)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if operation < 0 {
		return errUnsupportedKernel
	}

	partials := partialCount(count)

	var status C.CUDAStatus
	code := C.cuda_dispatch_pair_loss(
		contextRef,
		operation,
		elementFormat,
		predictionsBuffer,
		targetsBuffer,
		scratchBuffer,
		outBuffer,
		C.uint32_t(count),
		C.uint32_t(partials),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

/*
DispatchCrossEntropy launches row partial + finalize cross-entropy kernels on device.
*/
func DispatchCrossEntropy(
	contextRef C.CUDADeviceRef,
	logitsBuffer C.CUDABufferRef,
	targetsBuffer C.CUDABufferRef,
	scratchBuffer C.CUDABufferRef,
	outBuffer C.CUDABufferRef,
	errorFlagBuffer C.CUDABufferRef,
	format dtype.DType,
	batch uint32,
	classes uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_cross_entropy_loss(
		contextRef,
		elementFormat,
		logitsBuffer,
		targetsBuffer,
		scratchBuffer,
		outBuffer,
		errorFlagBuffer,
		C.uint32_t(batch),
		C.uint32_t(classes),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
