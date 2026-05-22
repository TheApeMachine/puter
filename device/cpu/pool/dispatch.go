package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
) {
	if batch*channels*inHeight*inWidth == 0 ||
		batch*channels*outHeight*outWidth == 0 {
		return
	}

	if format == dtype.Float32 {
		if useMax {
			runMaxPool2DF32(
				config, input, output,
				batch, channels, inHeight, inWidth, outHeight, outWidth,
				true,
			)

			return
		}

		runAvgPool2DF32(
			config, input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			false,
		)

		return
	}

	runPool2DReduced(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, useMax,
	)
}

func dispatchAdaptivePool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
) {
	if batch*channels*inHeight*inWidth == 0 ||
		batch*channels*outHeight*outWidth == 0 {
		return
	}

	if format == dtype.Float32 {
		if useMax {
			runAdaptiveMaxPool2DF32(
				input, output,
				batch, channels, inHeight, inWidth, outHeight, outWidth,
				true,
			)

			return
		}

		runAdaptiveAvgPool2DF32(
			input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			false,
		)

		return
	}

	runAdaptivePool2DReduced(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, useMax,
	)
}
