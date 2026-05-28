//go:build xla

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Each loss writes its scalar result into `*dst` (ARCHITECTURE.md §2.2).
The XLA host currently materializes the result through PJRT to a host
scalar and stores it at dst; once the planner / executable cache work
lands (GAPS.md §2.5–2.6) the result will be written directly into the
caller's PjRtBuffer slot.
*/
func (losses *Losses) MSE(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	losses.host.PairLossScalar(dst, predictions, targets, count, format, KernelMSE)
}

func (losses *Losses) MAE(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	losses.host.PairLossScalar(dst, predictions, targets, count, format, KernelMAE)
}

func (losses *Losses) Huber(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	losses.host.PairLossScalar(dst, predictions, targets, count, format, KernelHuber)
}

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
