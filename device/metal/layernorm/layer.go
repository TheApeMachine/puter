//go:build darwin && cgo

package layernorm

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (norm *Norm) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	norm.host.LaunchLayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func (norm *Norm) RMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	norm.host.LaunchRMSNorm(config, input, scale, output, rows, lastDim, format)
}
