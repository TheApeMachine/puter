//go:build xla

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) MSE(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count

	return losses.host.PairLossScalar(predictions, targets, format, KernelMSE)
}

func (losses *Losses) MAE(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count

	return losses.host.PairLossScalar(predictions, targets, format, KernelMAE)
}

func (losses *Losses) Huber(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count

	return losses.host.PairLossScalar(predictions, targets, format, KernelHuber)
}

func (losses *Losses) BinaryCrossEntropy(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count

	return losses.host.PairLossScalar(predictions, targets, format, KernelBinaryCrossEntropy)
}

func (losses *Losses) KLDivergence(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count

	return losses.host.PairLossScalar(predictions, targets, format, KernelKLDivergence)
}

func (losses *Losses) CrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	return losses.host.CrossEntropyScalar(logits, targets, batchSize, classes, format)
}
