//go:build cuda

package active_inference

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "belief.h"
#include "free_energy.h"
*/
import "C"

//go:embed active_inference.cuh
var activeInferenceHubSource string

//go:embed belief.cu
var beliefDomainSource string

//go:embed free_energy.cu
var freeEnergyDomainSource string

func moduleSource() string {
	parts := []string{
		activeInferenceHubSource,
		beliefDomainSource,
		freeEnergyDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_active_register_module_source(source)
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

func partialCount(count uint32) uint32 {
	return (count + 255) / 256
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA active inference dtype"}

func DispatchBeliefUpdate(
	contextRef C.CUDADeviceRef,
	likelihoodRef C.CUDABufferRef,
	priorRef C.CUDABufferRef,
	scratchRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	partials := partialCount(count)
	var status C.CUDAStatus
	code := C.cuda_dispatch_belief_update(
		contextRef,
		elementFormat,
		likelihoodRef,
		priorRef,
		scratchRef,
		outputRef,
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

func DispatchFreeEnergy(
	contextRef C.CUDADeviceRef,
	likelihoodRef C.CUDABufferRef,
	posteriorRef C.CUDABufferRef,
	priorRef C.CUDABufferRef,
	scratchRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	partials := partialCount(count)
	var status C.CUDAStatus
	code := C.cuda_dispatch_free_energy(
		contextRef,
		elementFormat,
		likelihoodRef,
		posteriorRef,
		priorRef,
		scratchRef,
		outputRef,
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

func DispatchExpectedFreeEnergy(
	contextRef C.CUDADeviceRef,
	predictedObsRef C.CUDABufferRef,
	preferredObsRef C.CUDABufferRef,
	predictedStateRef C.CUDABufferRef,
	scratchRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	obsCount uint32,
	stateCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	obsPartials := partialCount(obsCount)
	statePartials := partialCount(stateCount)
	var status C.CUDAStatus
	code := C.cuda_dispatch_expected_free_energy(
		contextRef,
		elementFormat,
		predictedObsRef,
		preferredObsRef,
		predictedStateRef,
		scratchRef,
		outputRef,
		C.uint32_t(obsCount),
		C.uint32_t(stateCount),
		C.uint32_t(obsPartials),
		C.uint32_t(statePartials),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchPrecisionWeight(
	contextRef C.CUDADeviceRef,
	errorsRef C.CUDABufferRef,
	precisionRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_precision_weight(
		contextRef,
		elementFormat,
		errorsRef,
		precisionRef,
		outputRef,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
