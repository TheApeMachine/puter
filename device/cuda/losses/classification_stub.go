//go:build !cuda

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (losses *Losses) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.stubHost()
	return 0
}

func (losses *Losses) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType,) float32 {
	losses.stubHost()
	return 0
}

func (losses *Losses) CrossEntropy(logits unsafe.Pointer, targets unsafe.Pointer, batchSize, classes int, format dtype.DType,) float32 {
	losses.stubHost()
	return 0
}
