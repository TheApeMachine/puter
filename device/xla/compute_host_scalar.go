//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/reduction"
)

func (host *ComputeHost) ReductionScalar(
	values unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel reduction.ReductionKernel,
) float32 {
	if count == 0 || host.bridge == nil {
		return 0
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format},
		InputShapes: []tensor.Shape{inputShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.borrowScalarBuffer(format)
	defer outputTensor.Close()

	inputTensor := host.requireDeviceTensor(values)
	operationName := reductionOperationName(kernel)

	host.dispatchError(host.builder.ExecuteReduction(
		host.bridge,
		operationName,
		context,
		inputTensor,
		outputTensor,
	))

	return host.readScalarFloat32(outputTensor)
}

func (host *ComputeHost) borrowScalarBuffer(format dtype.DType) *DeviceTensor {
	elementSize, err := format.Size()
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	bytesIn := make([]byte, elementSize)
	deviceTensor, err := host.bridge.stageUpload(scalarShape, format, bytesIn, false)
	host.dispatchError(err)

	return deviceTensor.(*DeviceTensor)
}

func (host *ComputeHost) readScalarFloat32(deviceTensor *DeviceTensor) float32 {
	_, bytesOut, err := host.bridge.download(deviceTensor)
	host.dispatchError(err)

	decoded, err := convert.BytesToFloat32(deviceTensor.format(), bytesOut)
	host.dispatchError(err)

	if len(decoded) == 0 {
		host.dispatchError(&loweringError{message: "empty XLA scalar download"})
	}

	return decoded[0]
}

func (host *ComputeHost) DotProduct(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 || host.bridge == nil {
		return 0
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{inputShape, inputShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.borrowScalarBuffer(format)
	defer outputTensor.Close()

	leftTensor := host.requireDeviceTensor(left)
	rightTensor := host.requireDeviceTensor(right)

	host.dispatchError(host.builder.ExecuteDot(
		host.bridge,
		context,
		leftTensor,
		rightTensor,
		outputTensor,
	))

	return host.readScalarFloat32(outputTensor)
}

func reductionOperationName(kernel reduction.ReductionKernel) string {
	switch kernel {
	case reduction.KernelSum:
		return "reduce_sum"
	case reduction.KernelProd:
		return "reduce_prod"
	case reduction.KernelMin:
		return "reduce_min"
	case reduction.KernelMax:
		return "reduce_max"
	case reduction.KernelL1Norm:
		return "reduce_l1norm"
	default:
		panic("xla: unsupported reduction kernel")
	}
}
