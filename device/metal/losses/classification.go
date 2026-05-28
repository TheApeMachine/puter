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
	losses.host.PairLossScalar(dst, predictions, targets, count, format, KernelBinaryCrossEntropy)
}

func (losses *Losses) KLDivergence(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	losses.host.PairLossScalar(dst, predictions, targets, count, format, KernelKLDivergence)
}

func (losses *Losses) CrossEntropy(
	dst unsafe.Pointer,
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) {
	losses.host.CrossEntropyScalar(dst, logits, targets, batchSize, classes, format)
}
