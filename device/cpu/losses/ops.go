package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/internal/scalar"
)

/*
Each loss writes its scalar result into `*dst`. Zero-host-sync per
ARCHITECTURE.md §2.2.
*/
func (losses Losses) MSE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchMSE(predictions, targets, count, format), format)
}

func (losses Losses) MAE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchMAE(predictions, targets, count, format), format)
}

func (losses Losses) Huber(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchHuber(predictions, targets, count, format), format)
}

func (losses Losses) BinaryCrossEntropy(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchBinaryCrossEntropy(predictions, targets, count, format), format)
}

func (losses Losses) KLDivergence(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	scalar.StoreFloat32(dst, dispatchKLDivergence(predictions, targets, count, format), format)
}

func (losses Losses) CrossEntropy(
	dst unsafe.Pointer,
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) {
	scalar.StoreFloat32(dst, dispatchCrossEntropy(logits, targets, batchSize, classes, format), format)
}
