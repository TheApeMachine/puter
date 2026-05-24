//go:build xla

package xla

/*
#cgo CXXFLAGS: -I${SRCDIR}/internal/bridge -std=c++17
#cgo LDFLAGS: -ldl -lstdc++

#include <stdlib.h>
#include "internal/bridge/core.h"
#include "internal/bridge/bridge_xla.cc"
*/
import "C"

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type xlaBridge struct {
	client   C.XLAClientRef
	backend  *Backend
	dtypes   []dtype.DType
	totalMem int64
}

func openXLABridge(backend *Backend) (*xlaBridge, error) {
	var client C.XLAClientRef
	var status C.XLAStatus
	code := C.xla_open_client(&client, &status)

	if code != 0 || client == nil {
		return nil, bridgeStatusError(status)
	}

	return &xlaBridge{
		client:   client,
		backend:  backend,
		dtypes:   SupportedDTypeSet(),
		totalMem: int64(C.xla_client_device_memory_bytes(client)),
	}, nil
}

func (bridge *xlaBridge) clientRef() C.XLAClientRef {
	return bridge.client
}

func (bridge *xlaBridge) supportedDTypes() []dtype.DType {
	return bridge.dtypes
}

func (bridge *xlaBridge) devicePoolBytes() int64 {
	return bridge.totalMem
}

func (bridge *xlaBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	tensorValue, err := bridge.uploadAsync(shape, sourceDType, bytesIn)

	if err != nil {
		return nil, err
	}

	if waitErr := tensorValue.(*DeviceTensor).WaitReady(); waitErr != nil {
		return nil, waitErr
	}

	return tensorValue, nil
}

func (bridge *xlaBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return bridge.stageUpload(shape, sourceDType, bytesIn, true)
}

func (bridge *xlaBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	deviceTensor, ok := input.(*DeviceTensor)

	if !ok {
		return dtype.Invalid, nil, tensor.ErrShapeMismatch
	}

	bytesOut := make([]byte, deviceTensor.byteCount)
	var status C.XLAStatus
	code := C.xla_buffer_to_host(
		bridge.client,
		deviceTensor.bufferRef(),
		unsafeBytes(bytesOut),
		C.longlong(deviceTensor.byteCount),
		&status,
	)

	if code != 0 {
		return dtype.Invalid, nil, bridgeStatusError(status)
	}

	return deviceTensor.format(), bytesOut, nil
}

func (bridge *xlaBridge) close() error {
	if bridge.client != nil {
		C.xla_close_client(bridge.client)
		bridge.client = nil
	}

	return nil
}

func (bridge *xlaBridge) stageUpload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
	async bool,
) (tensor.Tensor, error) {
	elementType, err := mapDTypeToBridge(sourceDType)

	if err != nil {
		return nil, err
	}

	dimensions := shapeToCInt64(shape)
	var status C.XLAStatus

	var dimensionPointer *C.longlong
	if len(dimensions) > 0 {
		dimensionPointer = (*C.longlong)(unsafe.Pointer(&dimensions[0]))
	}

	bufferRef := C.xla_buffer_from_host(
		bridge.client,
		unsafeBytes(bytesIn),
		C.longlong(len(bytesIn)),
		C.int(elementType),
		dimensionPointer,
		C.int(len(dimensions)),
		&status,
	)

	if bufferRef == nil {
		return nil, bridgeStatusError(status)
	}

	_ = async
	return newDeviceTensor(
		bridge.backend,
		shape,
		sourceDType,
		bufferRef,
		len(bytesIn),
		false,
	), nil
}

func (bridge *xlaBridge) compileHLO(hloText string) (C.XLAExecutableRef, error) {
	moduleText := C.CString(hloText)
	defer C.free(unsafe.Pointer(moduleText))

	var status C.XLAStatus
	executableRef := C.xla_compile_hlo(bridge.client, moduleText, &status)

	if executableRef == nil {
		return nil, bridgeStatusError(status)
	}

	return executableRef, nil
}

func (bridge *xlaBridge) executeUnary(
	executableRef C.XLAExecutableRef,
	input C.XLABufferRef,
	output C.XLABufferRef,
) error {
	var status C.XLAStatus
	code := C.xla_execute_unary(bridge.client, executableRef, input, output, &status)

	if code != 0 {
		return bridgeStatusError(status)
	}

	return nil
}

func (bridge *xlaBridge) executeBinary(
	executableRef C.XLAExecutableRef,
	left C.XLABufferRef,
	right C.XLABufferRef,
	output C.XLABufferRef,
) error {
	var status C.XLAStatus
	code := C.xla_execute_binary(bridge.client, executableRef, left, right, output, &status)

	if code != 0 {
		return bridgeStatusError(status)
	}

	return nil
}

func (bridge *xlaBridge) executeVariadic(
	executableRef C.XLAExecutableRef,
	inputs []*DeviceTensor,
	output *DeviceTensor,
) error {
	if len(inputs) == 0 {
		return &loweringError{message: "XLA variadic execute requires inputs"}
	}

	bufferRefs := make([]C.XLABufferRef, len(inputs))

	for inputIndex, inputTensor := range inputs {
		bufferRefs[inputIndex] = inputTensor.bufferRef()
	}

	var status C.XLAStatus
	code := C.xla_execute_variadic(
		bridge.client,
		executableRef,
		(*C.XLABufferRef)(unsafe.Pointer(&bufferRefs[0])),
		C.int(len(bufferRefs)),
		output.bufferRef(),
		&status,
	)

	if code != 0 {
		return bridgeStatusError(status)
	}

	return nil
}

func (bridge *xlaBridge) releaseBuffer(bufferRef C.XLABufferRef) {
	if bufferRef != nil {
		C.xla_buffer_release(bufferRef)
	}
}

func (bridge *xlaBridge) releaseExecutable(executableRef C.XLAExecutableRef) {
	if executableRef != nil {
		C.xla_executable_release(executableRef)
	}
}

func bridgeStatusError(status C.XLAStatus) error {
	if status.code == 0 {
		return nil
	}

	message := C.GoString(&status.message[0])
	return statusError(XLAStatus{Code: int(status.code), Message: message})
}

func mapDTypeToBridge(elementFormat dtype.DType) (int, error) {
	mapped, err := MapDType(elementFormat)

	if err != nil {
		return 0, err
	}

	switch mapped {
	case XLAElementF64:
		return 1, nil
	case XLAElementF32:
		return 2, nil
	case XLAElementF16:
		return 3, nil
	case XLAElementBF16:
		return 4, nil
	case XLAElementF8E4M3:
		return 5, nil
	case XLAElementF8E5M2:
		return 6, nil
	case XLAElementS64:
		return 7, nil
	case XLAElementS32:
		return 8, nil
	case XLAElementS16:
		return 9, nil
	case XLAElementS8:
		return 10, nil
	case XLAElementU64:
		return 11, nil
	case XLAElementU32:
		return 12, nil
	case XLAElementU16:
		return 13, nil
	case XLAElementU8:
		return 14, nil
	case XLAElementPred:
		return 15, nil
	default:
		return 0, errUnsupportedDType
	}
}

func shapeToCInt64(shape tensor.Shape) []int64 {
	dimensions := shape.Dims()
	output := make([]int64, len(dimensions))

	for index, dimension := range dimensions {
		output[index] = int64(dimension)
	}

	if len(output) == 0 {
		return []int64{1}
	}

	return output
}

func unsafeBytes(bytesIn []byte) unsafe.Pointer {
	if len(bytesIn) == 0 {
		return nil
	}

	return unsafe.Pointer(&bytesIn[0])
}
