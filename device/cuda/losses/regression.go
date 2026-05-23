//go:build cuda

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
