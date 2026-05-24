//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (host *ComputeHost) MatmulBiasGeluLaunch(
	out, left, right, bias unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
	if rows == 0 || inner == 0 || cols == 0 || host.bridge == nil {
		return
	}

	leftShape, rightShape, outputShape, err := ShapeFromMatmul(rows, inner, cols)
	host.dispatchError(err)

	biasShape, err := tensor.NewShape([]int{cols})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format},
		InputShapes: []tensor.Shape{leftShape, rightShape, biasShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	outputTensor := host.requireDeviceTensor(out)
	leftTensor := host.requireDeviceTensor(left)
	rightTensor := host.requireDeviceTensor(right)
	biasTensor := host.requireDeviceTensor(bias)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"matmul_bias_gelu",
		context,
		nil,
		nil,
		[]*DeviceTensor{leftTensor, rightTensor, biasTensor},
		outputTensor,
	))
}

func (host *ComputeHost) LayernormResidualLaunch(
	out, input, scale, bias, residual unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 || host.bridge == nil {
		return
	}

	inputShape, err := tensor.NewShape([]int{rows, lastDim})
	host.dispatchError(err)

	scaleShape, err := tensor.NewShape([]int{lastDim})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format, format, format},
		InputShapes: []tensor.Shape{inputShape, scaleShape, scaleShape, inputShape},
		OutputDType: format,
		OutputShape: inputShape,
	}

	outputTensor := host.requireDeviceTensor(out)
	inputTensor := host.requireDeviceTensor(input)
	scaleTensor := host.requireDeviceTensor(scale)
	biasTensor := host.requireDeviceTensor(bias)
	residualTensor := host.requireDeviceTensor(residual)

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"layernorm_residual",
		context,
		nil,
		nil,
		[]*DeviceTensor{inputTensor, scaleTensor, biasTensor, residualTensor},
		outputTensor,
	))
}
