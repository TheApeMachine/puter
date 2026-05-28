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
	dst unsafe.Pointer,
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel losses.LossKernel,
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

	predictionTensor := host.requireDeviceTensor(predictions)
	targetTensor := host.requireDeviceTensor(targets)
	outputTensor := host.requireDeviceTensor(dst)
	operationName := pairLossOperationName(kernel)

	host.dispatchError(host.builder.ExecutePairLoss(
		host.bridge,
		operationName,
		context,
		predictionTensor,
		targetTensor,
		outputTensor,
	))
}

func (host *ComputeHost) CrossEntropyScalar(
	dst unsafe.Pointer,
	logits, targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) {
	if batchSize == 0 || classes == 0 || host.bridge == nil {
		return
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

	logitsTensor := host.requireDeviceTensor(logits)
	targetTensor := host.requireDeviceTensor(targets)
	outputTensor := host.requireDeviceTensor(dst)

	host.dispatchError(host.builder.ExecuteCrossEntropy(
		host.bridge,
		context,
		logitsTensor,
		targetTensor,
		outputTensor,
	))
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
