//go:build xla

package normalization

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (normalization *Normalization) BatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType,) {
	normalization.unimplemented("BatchNormEval")
}

func (normalization *Normalization) GroupNorm(config device.GroupNormConfig, input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType,) {
	normalization.unimplemented("GroupNorm")
}

func (normalization *Normalization) InstanceNorm(input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType,) {
	normalization.unimplemented("InstanceNorm")
}

