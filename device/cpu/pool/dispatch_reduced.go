package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func runPool2DReduced(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
) {
	switch format {
	case dtype.BFloat16:
		Pool2DBFloat16Native(
			config, input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	case dtype.Float16:
		Pool2DFloat16Native(
			config, input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	default:
		Pool2DTypedScalar(
			format,
			config,
			input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	}
}

func runAdaptivePool2DReduced(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
) {
	switch format {
	case dtype.BFloat16:
		AdaptivePool2DBFloat16Native(
			input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	case dtype.Float16:
		AdaptivePool2DFloat16Native(
			input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	default:
		AdaptivePool2DTypedScalar(
			format,
			input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)
	}
}
