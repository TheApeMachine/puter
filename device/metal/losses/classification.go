//go:build darwin && cgo

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) BinaryCrossEntropy(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count

	*(*float32)(dst) = losses.host.PairLossScalar(predictions, targets, format, KernelBinaryCrossEntropy)
}

func (losses *Losses) KLDivergence(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count

	*(*float32)(dst) = losses.host.PairLossScalar(predictions, targets, format, KernelKLDivergence)
}

func (losses *Losses) CrossEntropy(
	dst unsafe.Pointer,
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) {
	*(*float32)(dst) = losses.host.CrossEntropyScalar(logits, targets, batchSize, classes, format)
}
