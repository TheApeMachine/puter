//go:build cuda

package predictive_coding

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
#include "forward.h"
#include "learning.h"
*/
import "C"

//go:embed predictive_coding.cuh
var predictiveCodingHubSource string

//go:embed forward.cu
var forwardDomainSource string

//go:embed learning.cu
var learningDomainSource string

func moduleSource() string {
	parts := []string{
		predictiveCodingHubSource,
		forwardDomainSource,
		learningDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_predictive_coding_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA predictive coding dtype"}

func DispatchPrediction(
	contextRef C.CUDADeviceRef,
	weightsRef C.CUDABufferRef,
	stateRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	outCount uint32,
	inCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_pc_prediction(
		contextRef,
		elementFormat,
		weightsRef,
		stateRef,
		outputRef,
		C.uint32_t(outCount),
		C.uint32_t(inCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchPredictionError(
	contextRef C.CUDADeviceRef,
	observedRef C.CUDABufferRef,
	predictedRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_pc_prediction_error(
		contextRef,
		elementFormat,
		observedRef,
		predictedRef,
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

func DispatchUpdateRepresentation(
	contextRef C.CUDADeviceRef,
	config device.PredictiveCodingConfig,
	weightsRef C.CUDABufferRef,
	stateRef C.CUDABufferRef,
	errorRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	outCount uint32,
	inCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_pc_update_representation(
		contextRef,
		elementFormat,
		weightsRef,
		stateRef,
		errorRef,
		outputRef,
		C.uint32_t(outCount),
		C.uint32_t(inCount),
		C.float(config.LearningRate),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchUpdateWeights(
	contextRef C.CUDADeviceRef,
	config device.PredictiveCodingConfig,
	weightsRef C.CUDABufferRef,
	stateRef C.CUDABufferRef,
	errorRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	outCount uint32,
	inCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_pc_update_weights(
		contextRef,
		elementFormat,
		weightsRef,
		stateRef,
		errorRef,
		outputRef,
		C.uint32_t(outCount),
		C.uint32_t(inCount),
		C.float(config.LearningRate),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
