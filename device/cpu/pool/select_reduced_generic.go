//go:build !arm64

package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Pool2DBFloat16Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	Pool2DTypedScalar(dtype.BFloat16, config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth, useMax)
}

func Pool2DFloat16Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	Pool2DTypedScalar(dtype.Float16, config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth, useMax)
}

func AdaptivePool2DBFloat16Native(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	AdaptivePool2DTypedScalar(dtype.BFloat16, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth, useMax)
}

func AdaptivePool2DFloat16Native(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	AdaptivePool2DTypedScalar(dtype.Float16, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth, useMax)
}
