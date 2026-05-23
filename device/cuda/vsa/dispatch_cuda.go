//go:build cuda

package vsa

import (
	_ "embed"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "vsa_dispatch.h"
*/
import "C"

//go:embed vsa.cuh
var vsaHubSource string

//go:embed vsa.cu
var vsaDomainSource string

func moduleSource() string {
	parts := []string{
		vsaHubSource,
		vsaDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_vsa_register_module_source(source)
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

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA VSA dtype"}

const (
	operationBind           C.int = 0
	operationBundle         C.int = 1
	operationPermute        C.int = 2
	operationInversePermute C.int = 3
)

func DispatchBind(
	contextRef C.CUDADeviceRef,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchBinary(contextRef, operationBind, leftBuffer, rightBuffer, outputBuffer, format, count)
}

func DispatchBundle(
	contextRef C.CUDADeviceRef,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchBinary(contextRef, operationBundle, leftBuffer, rightBuffer, outputBuffer, format, count)
}

func DispatchPermute(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchUnary(contextRef, operationPermute, inputBuffer, outputBuffer, format, count)
}

func DispatchInversePermute(
	contextRef C.CUDADeviceRef,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchUnary(contextRef, operationInversePermute, inputBuffer, outputBuffer, format, count)
}

func dispatchBinary(
	contextRef C.CUDADeviceRef,
	operation C.int,
	leftBuffer C.CUDABufferRef,
	rightBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_vsa_binary(
		contextRef,
		operation,
		elementFormat,
		leftBuffer,
		rightBuffer,
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

func dispatchUnary(
	contextRef C.CUDADeviceRef,
	operation C.int,
	inputBuffer C.CUDABufferRef,
	outputBuffer C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_vsa_unary(
		contextRef,
		operation,
		elementFormat,
		inputBuffer,
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
