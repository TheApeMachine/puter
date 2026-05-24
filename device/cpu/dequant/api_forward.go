package dequant

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

var defaultDequantization = New()

func Dequant(dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType) {
	defaultDequantization.Dequant(dst, src, count, config, dstFormat, srcFormat)
}

func Dequant4(dst, src unsafe.Pointer,
	pairCount int,
	config device.DequantInt4Config,
	dstFormat, srcFormat dtype.DType) {
	defaultDequantization.Dequant4(dst, src, pairCount, config, dstFormat, srcFormat)
}
