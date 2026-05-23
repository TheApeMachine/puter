//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (host *ComputeHost) MatmulLaunch(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
	if rows == 0 || inner == 0 || cols == 0 || host.bridge == nil {
		return
	}

	leftShape, rightShape, outputShape, err := ShapeFromMatmul(rows, inner, cols)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{leftShape, rightShape},
		OutputDType: format,
		OutputShape: outputShape,
	}

	leftTensor := host.requireDeviceTensor(left)
	rightTensor := host.requireDeviceTensor(right)
	outputTensor := host.requireDeviceTensor(out)

	host.dispatchError(host.builder.ExecuteMatmul(
		host.bridge,
		context,
		leftTensor,
		rightTensor,
		outputTensor,
	))
}
