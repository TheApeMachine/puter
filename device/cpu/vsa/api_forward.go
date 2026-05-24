package vsa

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultVSA = New()

func Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	defaultVSA.Bind(left, right, output, count, format)
}

func Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	defaultVSA.Bundle(left, right, output, count, format)
}

func InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	defaultVSA.InversePermute(config, input, output, count, format)
}

func Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	defaultVSA.Permute(config, input, output, count, format)
}

func Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	return defaultVSA.Similarity(left, right, count, format)
}
