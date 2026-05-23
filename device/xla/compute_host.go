//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/xla/activation"
	"github.com/theapemachine/puter/device/xla/elementwise"
)

/*
ComputeHost lowers and executes XLA programs for embedded operation families.
*/
type ComputeHost struct {
	bridge  *xlaBridge
	builder *Builder
}

func (host *ComputeHost) NeedsPlatform() {
	panic("xla: platform unavailable")
}

func (host *ComputeHost) dispatchError(err error) {
	if err != nil {
		panic(err)
	}
}

func (host *ComputeHost) elementCount(pointers ...unsafe.Pointer) int {
	for _, pointer := range pointers {
		deviceTensor := resolveDeviceTensor(pointer)

		if deviceTensor != nil {
			return deviceTensor.Len()
		}
	}

	return 0
}

func (host *ComputeHost) requireDeviceTensor(pointer unsafe.Pointer) *DeviceTensor {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor == nil {
		panic("xla: invalid resident tensor pointer")
	}

	return deviceTensor
}

func (host *ComputeHost) StandardUnary(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernel activation.StandardKernel,
) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	operationName, ok := activation.StandardKernelName(kernel)

	if !ok {
		panic("xla: unsupported activation kernel")
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, shape)
	input := host.requireDeviceTensor(src)
	output := host.requireDeviceTensor(dst)

	if input.format() != format || output.format() != format {
		panic("xla: dtype mismatch")
	}

	host.dispatchError(host.builder.ExecuteUnary(
		host.bridge,
		operationName,
		context,
		nil,
		nil,
		input,
		output,
	))
}

func (host *ComputeHost) UnaryElementwise(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernel elementwise.UnaryKernel,
) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	operationName, ok := elementwise.UnaryKernelName(kernel)

	if !ok {
		panic("xla: unsupported elementwise kernel")
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, shape)
	input := host.requireDeviceTensor(src)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteUnary(
		host.bridge,
		operationName,
		context,
		nil,
		nil,
		input,
		output,
	))
}

func (host *ComputeHost) BinaryElementwise(
	dst, left, right unsafe.Pointer,
	format dtype.DType,
	kernel elementwise.BinaryKernel,
) {
	count := host.elementCount(dst, left, right)

	if count == 0 {
		return
	}

	operationName, ok := elementwise.BinaryKernelName(kernel)

	if !ok {
		panic("xla: unsupported elementwise kernel")
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForBinary(format, shape, shape, shape)
	leftTensor := host.requireDeviceTensor(left)
	rightTensor := host.requireDeviceTensor(right)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteBinary(
		host.bridge,
		operationName,
		context,
		nil,
		nil,
		leftTensor,
		rightTensor,
		output,
	))
}

func (host *ComputeHost) UnaryParam(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernelName string,
	param float32,
) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, shape)
	input := host.requireDeviceTensor(src)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteUnary(
		host.bridge,
		kernelName,
		context,
		[]float64{float64(param)},
		nil,
		input,
		output,
	))
}

func (host *ComputeHost) NotImplemented(methodName string) {
	panic("xla: " + methodName + " not implemented")
}
