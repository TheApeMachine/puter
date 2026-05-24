//go:build xla

package layernorm

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (norm *Norm) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	norm.host.LaunchLayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func (norm *Norm) RMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	norm.host.LaunchRMSNorm(input, scale, output, rows, lastDim, format)
}
