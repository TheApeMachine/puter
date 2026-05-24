//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (host *ComputeHost) DispatchGrad1D(
	input, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"grad1d",
		context,
		[]float64{float64(physicsSpacingInverse(spacing, false))},
		nil,
		host.requireDeviceTensor(input),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchDivergence1D(
	input, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	host.DispatchGrad1D(input, output, count, spacing, format)
}

func (host *ComputeHost) DispatchLaplacian4(
	input, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"laplacian4",
		context,
		[]float64{float64(physicsLaplacian4InverseDenominator(spacing))},
		nil,
		host.requireDeviceTensor(input),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchLaplacian(
	input, output unsafe.Pointer,
	dims []int,
	spacing float32,
	format dtype.DType,
) {
	if len(dims) != 1 {
		host.dispatchError(&loweringError{message: "XLA laplacian currently supports rank-1 tensors"})
	}

	count := dims[0]

	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"laplacian1d",
		context,
		[]float64{float64(physicsSpacingInverse(spacing, true))},
		nil,
		host.requireDeviceTensor(input),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchQuantumPotential(
	density, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"quantum_potential",
		context,
		[]float64{float64(physicsSpacingInverse(spacing, true)), float64(physicsQuantumScale())},
		nil,
		host.requireDeviceTensor(density),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchBohmianVelocity(
	phase, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	inputShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContextForUnary(format, inputShape)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	scale := float32(1.0 / (2 * dxValue))

	host.dispatchError(host.builder.ExecuteResearchUnaryParam(
		host.bridge,
		"central_difference_interior",
		context,
		[]float64{float64(scale)},
		nil,
		host.requireDeviceTensor(phase),
		host.requireDeviceTensor(output),
	))
}

func (host *ComputeHost) DispatchMadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: vectorShape,
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"madelung_continuity",
		context,
		[]float64{float64(physicsSpacingInverse(spacing, false))},
		nil,
		[]*DeviceTensor{
			host.requireDeviceTensor(density),
			host.requireDeviceTensor(velocity),
		},
		host.requireDeviceTensor(residual),
	))
}

func (host *ComputeHost) DispatchFFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.dispatchFFT1D(realIn, imagIn, realOut, imagOut, count, format, false)
}

func (host *ComputeHost) DispatchIFFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.dispatchFFT1D(realIn, imagIn, realOut, imagOut, count, format, true)
}

func (host *ComputeHost) dispatchFFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType,
	inverse bool,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	vectorShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	stackShape, err := ShapeFromCount(count * 2)
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, format},
		InputShapes: []tensor.Shape{vectorShape, vectorShape},
		OutputDType: format,
		OutputShape: stackShape,
	}

	stackTensor := host.borrowVectorBuffer(format, count*2)
	defer stackTensor.Close()

	intParams := []int64{0}

	if inverse {
		intParams[0] = 1
	}

	host.dispatchError(host.builder.ExecuteVariadic(
		host.bridge,
		"fft1d",
		context,
		nil,
		intParams,
		[]*DeviceTensor{
			host.requireDeviceTensor(realIn),
			host.requireDeviceTensor(imagIn),
		},
		stackTensor,
	))

	singleShape, err := ShapeFromCount(count)
	host.dispatchError(err)

	realContext := LoweringContextForUnary(format, stackShape)
	realContext.OutputShape = singleShape

	host.dispatchError(host.builder.ExecuteVectorSliceCopy(
		host.bridge,
		realContext,
		0,
		count,
		stackTensor,
		host.requireDeviceTensor(realOut),
	))

	imagContext := LoweringContextForUnary(format, stackShape)
	imagContext.OutputShape = singleShape

	host.dispatchError(host.builder.ExecuteVectorSliceCopy(
		host.bridge,
		imagContext,
		count,
		count,
		stackTensor,
		host.requireDeviceTensor(imagOut),
	))
}

func (host *ComputeHost) borrowVectorBuffer(format dtype.DType, count int) *DeviceTensor {
	shape, err := ShapeFromCount(count)
	host.dispatchError(err)

	elementSize, err := format.Size()
	host.dispatchError(err)

	bytesIn := make([]byte, count*elementSize)
	deviceTensor, err := host.bridge.stageUpload(shape, format, bytesIn, false)
	host.dispatchError(err)

	return deviceTensor.(*DeviceTensor)
}
