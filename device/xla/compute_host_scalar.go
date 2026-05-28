//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/reduction"
)

func (host *ComputeHost) ReductionScalar(
	dst unsafe.Pointer,
	values unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel reduction.ReductionKernel,
) {
	if count == 0 || host.bridge == nil {
		return
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

	inputTensor := host.requireDeviceTensor(values)
	outputTensor := host.requireDeviceTensor(dst)
	operationName := reductionOperationName(kernel)

	host.dispatchError(host.builder.ExecuteReduction(
		host.bridge,
		operationName,
		context,
		inputTensor,
		outputTensor,
	))
}

func (host *ComputeHost) DotProduct(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
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

	leftTensor := host.requireDeviceTensor(left)
	rightTensor := host.requireDeviceTensor(right)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteDot(
		host.bridge,
		context,
		leftTensor,
		rightTensor,
		outputTensor,
	))
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
