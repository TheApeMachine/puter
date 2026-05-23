//go:build darwin && cgo

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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
