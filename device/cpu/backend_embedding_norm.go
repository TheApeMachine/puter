package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/embedding"
	"github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/cpu/normalization"
)

func (backend *Backend) Lookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	embedding.Lookup(table, indices, output, vocab, hidden, indexCount, format)
}

func (backend *Backend) Bag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	embedding.Bag(table, indices, offsets, output, vocab, hidden, bagCount, indexCount, format)
}

func (backend *Backend) GroupNorm(
	config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	normalization.GroupNorm(groupNormConfig(config), input, scale, bias, output, batch, channels, spatial, format)
}

func (backend *Backend) InstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	normalization.InstanceNorm(input, scale, bias, output, batch, channels, spatial, format)
}

func (backend *Backend) BatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	normalization.BatchNormEval(input, scale, bias, mean, variance, output, batch, channels, spatial, format)
}

func (backend *Backend) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	layernorm.LayerNorm(input, scale, bias, output, rows, lastDim, format)
}

func (backend *Backend) RMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	layernorm.RMSNorm(input, scale, output, rows, lastDim, format)
}
