//go:build !cuda

package dequant

import (
	"unsafe"
	
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (dequantization *Dequantization) Dequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	dequantization.stubHost()
}

func (dequantization *Dequantization) Dequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	dequantization.stubHost()
}

