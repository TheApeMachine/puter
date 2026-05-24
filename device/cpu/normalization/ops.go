package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (normalization Normalization) GroupNorm(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchGroupNorm(config, input, scale, bias, output, batch, channels, spatial, format)
}

func (normalization Normalization) InstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchInstanceNorm(input, scale, bias, output, batch, channels, spatial, format)
}

func (normalization Normalization) BatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	dispatchBatchNormEval(input, scale, bias, mean, variance, output, batch, channels, spatial, format)
}
