//go:build cuda

package losses

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Each regression loss writes its scalar result into `*dst`
(ARCHITECTURE.md §2.2). The CUDA host currently computes on device and
reads back internally; once the static memory planner lands
(GAPS.md P1) the host signature will take a device pointer directly.
*/
func (losses *Losses) MSE(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count

	*(*float32)(dst) = losses.host.PairLossScalar(predictions, targets, format, KernelMSE)
}

func (losses *Losses) MAE(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count

	*(*float32)(dst) = losses.host.PairLossScalar(predictions, targets, format, KernelMAE)
}

func (losses *Losses) Huber(
	dst, predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count

	*(*float32)(dst) = losses.host.PairLossScalar(predictions, targets, format, KernelHuber)
}
