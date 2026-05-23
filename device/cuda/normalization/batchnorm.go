//go:build cuda

package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (normalization *Normalization) BatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	normalization.host.DispatchBatchNormEval(input, scale, bias, mean, variance, output, batch, channels, spatial, format)
}
