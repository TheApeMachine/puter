package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/attention"
	"github.com/theapemachine/puter/device/cpu/vsa"
)

func (backend *Backend) ScaledDotProductAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	attention.ScaledDotProductAttention(
		flashAttentionConfig(config), query, key, value, output,
		seqQ, seqK, depth, valueDim, format,
	)
}

func (backend *Backend) FlashAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
) {
	attention.FlashAttention(
		flashAttentionConfig(config), query, key, value, output,
		seqQ, seqK, depth, valueDim, format,
	)
}

func (backend *Backend) MultiHeadAttention(
	config device.MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	attention.MultiHeadAttention(
		multiHeadAttentionConfig(config), query, key, value, output,
		seqQ, seqK, format,
	)
}

func (backend *Backend) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.Bind(left, right, output, count, format)
}

func (backend *Backend) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.Bundle(left, right, output, count, format)
}

func (backend *Backend) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.Permute(vsaConfig(config), input, output, count, format)
}

func (backend *Backend) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	vsa.InversePermute(vsaConfig(config), input, output, count, format)
}

func (backend *Backend) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return vsa.Similarity(left, right, count, format)
}
