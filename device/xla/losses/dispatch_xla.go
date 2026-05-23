//go:build xla

package losses

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.unimplemented("MSE")
	return 0
}

func (losses *Losses) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.unimplemented("MAE")
	return 0
}

func (losses *Losses) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.unimplemented("Huber")
	return 0
}

func (losses *Losses) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.unimplemented("BinaryCrossEntropy")
	return 0
}

func (losses *Losses) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.unimplemented("KLDivergence")
	return 0
}

func (losses *Losses) CrossEntropy(logits unsafe.Pointer, targets unsafe.Pointer, batchSize, classes int, format dtype.DType,) float32 {
	losses.unimplemented("CrossEntropy")
	return 0
}

