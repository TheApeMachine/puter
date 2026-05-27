package layernorm

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (norm Norm) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	dispatchLayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func (norm Norm) RMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	dispatchRMSNorm(config, input, scale, output, rows, lastDim, format)
}

func (norm Norm) ModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	dispatchModulatedLayerNorm(
		config,
		input, modulation, output,
		rows, lastDim, rowsPerBatch, modulationCols,
		format,
	)
}
