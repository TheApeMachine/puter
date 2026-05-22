//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include <stdlib.h>
#include "bridge_darwin.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func runMetalUnaryParam(
	kernelName string,
	input tensor.Tensor,
	out tensor.Tensor,
	param float32,
) error {
	inputTensor, outTensor, err := requireUnaryElementwiseTensors(input, out)

	if err != nil {
		return err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)

	if err != nil {
		return err
	}

	if inputTensor.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor)

	if err != nil {
		return err
	}

	kernelFullName := kernelName + "_float32"

	status := C.MetalStatus{}
	kernelCString := C.CString(kernelFullName)
	defer C.free(unsafe.Pointer(kernelCString))

	rc := C.metal_dispatch_unary_param(
		inputTensor.bridge.device,
		kernelCString,
		C.int(elementDType),
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(inputTensor.shape.Len()),
		C.float(param),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal unary param %s: %s", kernelName, metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)

		return err
	}

	return nil
}

func runMetalAxpy(y tensor.Tensor, x tensor.Tensor, alpha float32) error {
	yTensor, err := requireMetalTensor(y)

	if err != nil {
		return err
	}

	xTensor, err := requireMetalTensor(x)

	if err != nil {
		return err
	}

	if yTensor.dtype != xTensor.dtype {
		return tensor.ErrDTypeMismatch
	}

	if !yTensor.shape.Equal(xTensor.shape) {
		return tensor.ErrShapeMismatch
	}

	if yTensor.shape.Len() == 0 {
		return nil
	}

	elementDType, err := metalElementDTypeFor(yTensor.dtype)

	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(yTensor, xTensor)

	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_axpy(
		yTensor.bridge.device,
		C.int(elementDType),
		yTensor.buffer,
		xTensor.buffer,
		C.uint32_t(yTensor.shape.Len()),
		C.float(alpha),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal axpy: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)

		return err
	}

	return nil
}

func runMetalDot(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	leftTensor, rightTensor, outTensor, err := requireMetalDotTensors(left, right, out)

	if err != nil {
		return err
	}

	if leftTensor.shape.Len() == 0 {
		return nil
	}

	elementDType, err := metalElementDTypeFor(leftTensor.dtype)

	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(outTensor, leftTensor, rightTensor)

	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_dot(
		leftTensor.bridge.device,
		C.int(elementDType),
		leftTensor.buffer,
		rightTensor.buffer,
		outTensor.buffer,
		C.uint32_t(leftTensor.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal dot: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)

		return err
	}

	return nil
}
