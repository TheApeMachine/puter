//go:build cuda

package dropout

import (
	_ "embed"
	"math"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
#include "dropout.h"
#include "mask.h"
*/
import "C"

//go:embed dropout.cuh
var dropoutHubSource string

//go:embed mask.cu
var maskDomainSource string

func moduleSource() string {
	parts := []string{
		dropoutHubSource,
		maskDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_dropout_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA dropout dtype"}

func dropoutSeedState(seed uint64) (seedX, seedY, seedZ, seedW uint32) {
	seedX = uint32(seed)
	seedY = uint32(seed >> 32)
	seedZ = uint32(seed ^ 0x9e3779b9)
	seedW = uint32((seed >> 32) ^ 0x6c078965)

	return seedX, seedY, seedZ, seedW
}

func dropoutThreshold(keepProb float32) uint32 {
	return uint32(float64(keepProb) * (1 << 32))
}

func DispatchDropout(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	count uint32,
	config device.DropoutConfig,
	format dtype.DType,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	identity := 0

	if config.Rate <= 0 {
		identity = 1
	}

	keepProb := float32(1.0 - config.Rate)
	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)
	seedX, seedY, seedZ, seedW := dropoutSeedState(config.Seed)

	var status C.CUDAStatus
	code := C.cuda_dispatch_dropout(
		contextRef,
		elementFormat,
		inputBuffer,
		outputBuffer,
		C.uint32_t(count),
		C.float(scale),
		C.uint32_t(threshold),
		C.uint32_t(seedX),
		C.uint32_t(seedY),
		C.uint32_t(seedZ),
		C.uint32_t(seedW),
		C.int(identity),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	if identity == 0 && math.IsInf(float64(scale), 1) {
		return &dispatchError{code: -6, message: "invalid CUDA dropout rate"}
	}

	return nil
}
