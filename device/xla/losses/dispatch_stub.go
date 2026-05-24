//go:build !xla

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) MSE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}

func (losses *Losses) MAE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}

func (losses *Losses) Huber(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}

func (losses *Losses) BinaryCrossEntropy(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}

func (losses *Losses) KLDivergence(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}

func (losses *Losses) CrossEntropy(dst unsafe.Pointer, logits unsafe.Pointer, targets unsafe.Pointer, batchSize, classes int, format dtype.DType) {
	losses.stubHost()
	*(*float32)(dst) = 0
}
