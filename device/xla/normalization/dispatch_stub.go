//go:build !xla

package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (normalization *Normalization) GroupNorm( config GroupNormConfig, input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType, ) {
	normalization.stubHost()
}

func (normalization *Normalization) InstanceNorm( input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType, ) {
	normalization.stubHost()
}

func (normalization *Normalization) BatchNormEval( input, scale, bias, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType, ) {
	normalization.stubHost()
}

