package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func MaxPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchPool2D(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, true,
	)
}

func AvgPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchPool2D(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, false,
	)
}

func AdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchAdaptivePool2D(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, true,
	)
}

func AdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchAdaptivePool2D(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, false,
	)
}
