//go:build cuda

package causal

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "causal_dispatch.h"
*/
import "C"

//go:embed causal.cuh
var causalHubSource string

//go:embed adjustment.cu
var adjustmentSource string

//go:embed intervention.cu
var interventionSource string

//go:embed dag.cu
var dagSource string

//go:embed matrix.cuh
var matrixHubSource string

//go:embed matrix.cu
var matrixSource string

func moduleSource() string {
	parts := []string{
		causalHubSource,
		matrixHubSource,
		adjustmentSource,
		interventionSource,
		dagSource,
		matrixSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_causal_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA causal dtype"}

func partialCount(count uint32) uint32 {
	return (count + 255) / 256
}

func ivEstimateScratchBytes(partialCount uint32) int64 {
	return int64(partialCount) * 5 * 4
}

func dagMarkovScratchBytes(partialCount uint32) int64 {
	return int64(partialCount) * 4
}

func DispatchBackdoor(
	contextRef C.CUDADeviceRef,
	conditionalBuffer C.CUDABufferRef,
	marginalBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	xCount uint32,
	zCount uint32,
	yCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if xCount == 0 || zCount == 0 || yCount == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_backdoor_adjustment(
		contextRef,
		elementFormat,
		conditionalBuffer,
		marginalBuffer,
		outputBuffer,
		C.uint32_t(xCount),
		C.uint32_t(zCount),
		C.uint32_t(yCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchFrontdoor(
	contextRef C.CUDADeviceRef,
	mediatorBuffer C.CUDABufferRef,
	outcomeBuffer C.CUDABufferRef,
	marginalBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	xCount uint32,
	mCount uint32,
	yCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if xCount == 0 || mCount == 0 || yCount == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_frontdoor_adjustment(
		contextRef,
		elementFormat,
		mediatorBuffer,
		outcomeBuffer,
		marginalBuffer,
		outputBuffer,
		C.uint32_t(xCount),
		C.uint32_t(mCount),
		C.uint32_t(yCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchDoIntervene(
	contextRef C.CUDADeviceRef,
	adjacencyBuffer C.CUDABufferRef,
	intervenedBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	nodeCount uint32,
	intervenedCount uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if nodeCount == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_do_intervene(
		contextRef,
		elementFormat,
		adjacencyBuffer,
		intervenedBuffer,
		outputBuffer,
		C.uint32_t(nodeCount),
		C.uint32_t(intervenedCount),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchCATE(
	contextRef C.CUDADeviceRef,
	treatedBuffer C.CUDABufferRef,
	controlBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if count == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_cate(
		contextRef,
		elementFormat,
		treatedBuffer,
		controlBuffer,
		outputBuffer,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchCounterfactual(
	contextRef C.CUDADeviceRef,
	observedYBuffer C.CUDABufferRef,
	observedXBuffer C.CUDABufferRef,
	counterfactualXBuffer C.CUDABufferRef,
	slopeBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if count == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_counterfactual(
		contextRef,
		elementFormat,
		observedYBuffer,
		observedXBuffer,
		counterfactualXBuffer,
		slopeBuffer,
		outputBuffer,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchIVEstimate(
	contextRef C.CUDADeviceRef,
	instrumentBuffer C.CUDABufferRef,
	treatmentBuffer C.CUDABufferRef,
	outcomeBuffer C.CUDABufferRef,
	scratchBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if count == 0 {
		return nil
	}

	partials := partialCount(count)

	var status C.CUDAStatus
	code := C.cuda_dispatch_iv_estimate(
		contextRef,
		elementFormat,
		instrumentBuffer,
		treatmentBuffer,
		outcomeBuffer,
		scratchBuffer,
		outputBuffer,
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

func DispatchDAGMarkovFactorization(
	contextRef C.CUDADeviceRef,
	conditionalsBuffer C.CUDABufferRef,
	scratchBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if count == 0 {
		return nil
	}

	partials := partialCount(count)

	var status C.CUDAStatus
	code := C.cuda_dispatch_dag_markov_factorization(
		contextRef,
		elementFormat,
		conditionalsBuffer,
		nil,
		scratchBuffer,
		outputBuffer,
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

func DispatchCholesky(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	matrixOrder uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	if matrixOrder == 0 {
		return nil
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_cholesky(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.uint32_t(matrixOrder),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}
