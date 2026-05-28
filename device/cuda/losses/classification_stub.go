//go:build !cuda

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) BinaryCrossEntropy(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
}

func (losses *Losses) KLDivergence(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType) {
	losses.stubHost()
}

func (losses *Losses) CrossEntropy(dst unsafe.Pointer, logits unsafe.Pointer, targets unsafe.Pointer, batchSize, classes int, format dtype.DType) {
	losses.stubHost()
}
