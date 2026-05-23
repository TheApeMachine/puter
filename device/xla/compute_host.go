//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
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

func (host *ComputeHost) matrixRowsCols(pointer unsafe.Pointer) (rows int, cols int) {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor == nil {
		return 0, 0
	}

	shape := deviceTensor.Shape().Dims()

	if len(shape) == 0 {
		return 0, 0
	}

	cols = shape[len(shape)-1]
	total := deviceTensor.Len()

	if cols == 0 {
		return 0, 0
	}

	return total / cols, cols
}

func (host *ComputeHost) DualParam(
	dst, src unsafe.Pointer,
	format dtype.DType,
	kernelName string,
	param0, param1 float32,
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
		[]float64{float64(param0), float64(param1)},
		nil,
		input,
		output,
	))
}

func (host *ComputeHost) Softmax(dst, src unsafe.Pointer, format dtype.DType) {
	rows, cols := host.matrixRowsCols(src)

	if rows == 0 || cols == 0 {
		return
	}

	outputShape, err := tensor.NewShape([]int{rows, cols})
	host.dispatchError(err)

	context := LoweringContextForUnary(format, outputShape)
	input := host.requireDeviceTensor(src)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteUnary(
		host.bridge,
		"softmax",
		context,
		nil,
		nil,
		input,
		output,
	))
}

func (host *ComputeHost) PReLUV(dst, src, slopes unsafe.Pointer, format dtype.DType) {
	count := host.elementCount(dst, src, slopes)

	if count == 0 {
		return
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForBinary(format, shape, shape, shape)
	sourceTensor := host.requireDeviceTensor(src)
	slopeTensor := host.requireDeviceTensor(slopes)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteBinary(
		host.bridge,
		"prelu_v",
		context,
		nil,
		nil,
		sourceTensor,
		slopeTensor,
		output,
	))
}

func (host *ComputeHost) GLUPacked(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
	variant activation.GLUVariant,
) {
	if batch == 0 || halfCount == 0 {
		return
	}

	outputShape, err := tensor.NewShape([]int{batch, halfCount})
	host.dispatchError(err)

	inputShape, err := tensor.NewShape([]int{batch, halfCount * 2})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	operationName, ok := activation.GLUVariantName(variant)

	if !ok {
		panic("xla: unsupported GLU variant")
	}

	packedTensor := host.requireDeviceTensor(packed)
	output := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteUnary(
		host.bridge,
		operationName,
		context,
		nil,
		[]int64{int64(batch), int64(halfCount)},
		packedTensor,
		output,
	))
}

func (host *ComputeHost) GLUTensors(
	dst, gate, up unsafe.Pointer,
	format dtype.DType,
	variant activation.GLUVariant,
) {
	count := host.elementCount(dst, gate, up)

	if count == 0 {
		return
	}

	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForBinary(format, shape, shape, shape)
	gateTensor := host.requireDeviceTensor(gate)
	upTensor := host.requireDeviceTensor(up)
	output := host.requireDeviceTensor(dst)

	operationName, ok := activation.GLUVariantName(variant)

	if !ok {
		panic("xla: unsupported GLU variant")
	}

	host.dispatchError(host.builder.ExecuteBinary(
		host.bridge,
		operationName,
		context,
		nil,
		nil,
		gateTensor,
		upTensor,
		output,
	))
}
