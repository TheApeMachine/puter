//go:build !cuda

package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (normalization *Normalization) InstanceNorm(input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType,) {
	normalization.stubHost()
}
