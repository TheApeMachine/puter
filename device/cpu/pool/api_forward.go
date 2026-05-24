package pool

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultPool = New()

func AdaptiveAvgPool2D(input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	defaultPool.AdaptiveAvgPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func AdaptiveMaxPool2D(input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	defaultPool.AdaptiveMaxPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func AvgPool2D(config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	defaultPool.AvgPool2D(config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func MaxPool2D(config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	defaultPool.MaxPool2D(config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}
