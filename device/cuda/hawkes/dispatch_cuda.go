//go:build cuda

package hawkes

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "hawkes_dispatch.h"
*/
import "C"

//go:embed hawkes.cuh
var hawkesHubSource string

//go:embed intensity.cu
var intensityDomainSource string

//go:embed kernel.cu
var kernelDomainSource string

//go:embed likelihood.cu
var likelihoodDomainSource string

//go:embed markov.cu
var markovDomainSource string

func moduleSource() string {
	parts := []string{
		hawkesHubSource,
		intensityDomainSource,
		kernelDomainSource,
		likelihoodDomainSource,
		markovDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_hawkes_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA Hawkes dtype"}

func partialReductionCount(count uint32) uint32 {
	return (count + 255) / 256
}

func DispatchHawkesIntensity(
	contextRef C.CUDADeviceRef,
	eventsRef C.CUDABufferRef,
	queryTimesRef C.CUDABufferRef,
	baselineRef C.CUDABufferRef,
	alphaRef C.CUDABufferRef,
	betaRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	eventCount uint32,
	queryCount uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_hawkes_intensity(
		contextRef,
		elementFormat,
		eventsRef,
		queryTimesRef,
		baselineRef,
		alphaRef,
		betaRef,
		outputRef,
		C.uint32_t(eventCount),
		C.uint32_t(queryCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchHawkesKernelMatrix(
	contextRef C.CUDADeviceRef,
	eventsRef C.CUDABufferRef,
	alphaRef C.CUDABufferRef,
	betaRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	eventCount uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_hawkes_kernel_matrix(
		contextRef,
		elementFormat,
		eventsRef,
		alphaRef,
		betaRef,
		outputRef,
		C.uint32_t(eventCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchHawkesLogLikelihood(
	contextRef C.CUDADeviceRef,
	eventsRef C.CUDABufferRef,
	totalTimeRef C.CUDABufferRef,
	baselineRef C.CUDABufferRef,
	alphaRef C.CUDABufferRef,
	betaRef C.CUDABufferRef,
	scratchRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	eventCount uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_hawkes_log_likelihood(
		contextRef,
		elementFormat,
		eventsRef,
		totalTimeRef,
		baselineRef,
		alphaRef,
		betaRef,
		scratchRef,
		outputRef,
		C.uint32_t(eventCount),
		C.uint32_t(partialReductionCount(eventCount)),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchMarkovBlanketPartition(
	contextRef C.CUDADeviceRef,
	adjacencyRef C.CUDABufferRef,
	internalRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	nodeCount uint32,
	internalCount uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_markov_blanket_partition(
		contextRef,
		elementFormat,
		adjacencyRef,
		internalRef,
		outputRef,
		C.uint32_t(nodeCount),
		C.uint32_t(internalCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchMarkovFlow(
	contextRef C.CUDADeviceRef,
	mutualInformationRef C.CUDABufferRef,
	partitionRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	nodeCount uint32,
	targetLabel int32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_markov_flow(
		contextRef,
		elementFormat,
		mutualInformationRef,
		partitionRef,
		outputRef,
		C.uint32_t(nodeCount),
		C.int32_t(targetLabel),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchMarkovMutualInformation(
	contextRef C.CUDADeviceRef,
	jointRef C.CUDABufferRef,
	scratchRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	rows uint32,
	cols uint32,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	total := rows * cols
	var status C.CUDAStatus
	code := C.cuda_dispatch_markov_mutual_information(
		contextRef,
		elementFormat,
		jointRef,
		scratchRef,
		outputRef,
		C.uint32_t(rows),
		C.uint32_t(cols),
		C.uint32_t(partialReductionCount(total)),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func HawkesLogLikelihoodScratchBytes(eventCount uint32) int64 {
	return int64(eventCount) * 4
}

func MarkovMutualInformationScratchBytes(rows, cols uint32) int64 {
	return int64(partialReductionCount(rows*cols)) * 4
}

func ScalarScratchBytes(format dtype.DType) int64 {
	switch format {
	case dtype.Float32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 4
	}
}
