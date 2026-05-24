package normalization

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultNormalization = New()

func BatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	defaultNormalization.BatchNormEval(input, scale, bias, mean, variance, output, batch, channels, spatial, format)
}

func GroupNorm(config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	defaultNormalization.GroupNorm(config, input, scale, bias, output, batch, channels, spatial, format)
}

func InstanceNorm(input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	defaultNormalization.InstanceNorm(input, scale, bias, output, batch, channels, spatial, format)
}
