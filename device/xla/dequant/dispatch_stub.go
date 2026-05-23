//go:build !xla

package dequant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (dequantization *Dequantization) Dequant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	dequantization.stubHost()
}

func (dequantization *Dequantization) Dequant4(dst, src unsafe.Pointer, pairCount int, config DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	dequantization.stubHost()
}

