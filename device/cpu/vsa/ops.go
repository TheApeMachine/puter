package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (vsa VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBindFloat32Native(outputView, leftView, rightView)
}

func (vsa VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)
	outputView := unsafe.Slice((*float32)(output), count)

	VsaBundleFloat32Native(outputView, leftView, rightView)
}

func (vsa VSA) Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	requireVSAFloat32(format)

	inputView := unsafe.Slice((*float32)(input), count)
	outputView := unsafe.Slice((*float32)(output), count)
	shift := config.Shift % count

	if shift < 0 {
		shift += count
	}

	VsaPermuteFloat32Native(outputView, inputView, shift)
}

func (vsa VSA) InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	inverted := config
	inverted.Shift = -config.Shift
	vsa.Permute(inverted, input, output, count, format)
}

func (vsa VSA) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	requireVSAFloat32(format)

	leftView := unsafe.Slice((*float32)(left), count)
	rightView := unsafe.Slice((*float32)(right), count)

	return VsaSimilarityFloat32Native(leftView, rightView)
}

func requireVSAFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("vsa: unsupported dtype")
	}
}
