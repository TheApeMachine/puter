package quant

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

var defaultQuantization = New()

func Quant(dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType) {
	defaultQuantization.Quant(dst, src, count, config, dstFormat, srcFormat)
}
