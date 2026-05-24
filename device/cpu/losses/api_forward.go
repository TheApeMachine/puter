package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultLosses = New()

func MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultLosses.MSE(predictions, targets, count, format)
}

func MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultLosses.MAE(predictions, targets, count, format)
}

func Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultLosses.Huber(predictions, targets, count, format)
}

func BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultLosses.BinaryCrossEntropy(predictions, targets, count, format)
}

func KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultLosses.KLDivergence(predictions, targets, count, format)
}

func CrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	return defaultLosses.CrossEntropy(logits, targets, batchSize, classes, format)
}
