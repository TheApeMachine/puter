//go:build darwin && cgo

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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
