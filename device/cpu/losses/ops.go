package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses Losses) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchMSE(predictions, targets, count, format)
}

func (losses Losses) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchMAE(predictions, targets, count, format)
}

func (losses Losses) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchHuber(predictions, targets, count, format)
}

func (losses Losses) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchBinaryCrossEntropy(predictions, targets, count, format)
}

func (losses Losses) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchKLDivergence(predictions, targets, count, format)
}

func (losses Losses) CrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	return dispatchCrossEntropy(logits, targets, batchSize, classes, format)
}
