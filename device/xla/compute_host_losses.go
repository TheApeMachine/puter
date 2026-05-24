//go:build xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/internal/hlo"
	"github.com/theapemachine/puter/device/xla/losses"
)

func (host *ComputeHost) PairLossScalar(
	predictions, targets unsafe.Pointer,
	format dtype.DType,
	kernel losses.LossKernel,
) float32 {
	count := host.elementCount(predictions, targets)

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

	predictionTensor := host.requireDeviceTensor(predictions)
	targetTensor := host.requireDeviceTensor(targets)
	operationName := pairLossOperationName(kernel)

	host.dispatchError(host.builder.ExecutePairLoss(
		host.bridge,
		operationName,
		context,
		predictionTensor,
		targetTensor,
		outputTensor,
	))

	return host.readScalarFloat32(outputTensor)
}

func (host *ComputeHost) CrossEntropyScalar(
	logits, targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	if batchSize == 0 || classes == 0 || host.bridge == nil {
		return 0
	}

	logitsShape, err := ShapeFromRowsCols(batchSize, classes)
	host.dispatchError(err)

	targetShape, err := ShapeFromVector(batchSize)
	host.dispatchError(err)

	scalarShape, err := tensor.NewShape([]int{})
	host.dispatchError(err)

	context := LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{format, dtype.Int32},
		InputShapes: []tensor.Shape{logitsShape, targetShape},
		OutputDType: format,
		OutputShape: scalarShape,
	}

	outputTensor := host.borrowScalarBuffer(format)
	defer outputTensor.Close()

	logitsTensor := host.requireDeviceTensor(logits)
	targetTensor := host.requireDeviceTensor(targets)

	host.dispatchError(host.builder.ExecuteCrossEntropy(
		host.bridge,
		context,
		logitsTensor,
		targetTensor,
		outputTensor,
	))

	return host.readScalarFloat32(outputTensor)
}

func pairLossOperationName(kernel losses.LossKernel) string {
	switch kernel {
	case losses.KernelMSE:
		return hlo.PairLossOperationName("mse")
	case losses.KernelMAE:
		return hlo.PairLossOperationName("mae")
	case losses.KernelHuber:
		return hlo.PairLossOperationName("huber")
	case losses.KernelBinaryCrossEntropy:
		return hlo.PairLossOperationName("bce")
	case losses.KernelKLDivergence:
		return hlo.PairLossOperationName("kl")
	default:
		panic("xla: unsupported loss kernel")
	}
}
