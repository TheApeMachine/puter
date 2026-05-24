package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (pool Pool) MaxPool2D(
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

func (pool Pool) AvgPool2D(
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

func (pool Pool) AdaptiveMaxPool2D(
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

func (pool Pool) AdaptiveAvgPool2D(
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
