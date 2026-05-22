//go:build arm64

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
	if poolConfigNEONEligible(config) {
		pool2DBFloat16FastRowNative(
			config, input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)

		return
	}

	Pool2DTypedScalar(
		dtype.BFloat16,
		config,
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func Pool2DFloat16Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	if poolConfigNEONEligible(config) {
		pool2DFloat16FastRowNative(
			config, input, output,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)

		return
	}

	Pool2DTypedScalar(
		dtype.Float16,
		config,
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func pool2DBFloat16FastRowNative(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	loadInput, storeOutput := elementAccessors(dtype.BFloat16)

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				blockCols := outWidth &^ 3

				if blockCols > 0 {
					if useMax {
						MaxPool2DStride1RowBF16NEONAsm(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth,
							outRow*config.StrideH-config.PaddingH,
						)
					}

					if !useMax {
						AvgPool2DStride1RowBF16NEONAsm(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth,
							outRow*config.StrideH-config.PaddingH,
						)
					}
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					value := poolWindowTyped(
						input, channelOffsetIn,
						loadInput,
						inHeight, inWidth,
						outRow, outCol,
						config, useMax,
					)

					storeOutput(output, channelOffsetOut+outRow*outWidth+outCol, value)
				}
			}
		}
	}
}

func pool2DFloat16FastRowNative(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	loadInput, storeOutput := elementAccessors(dtype.Float16)

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				blockCols := outWidth &^ 3

				if blockCols > 0 {
					if useMax {
						MaxPool2DStride1RowFP16NEONAsm(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth,
							outRow*config.StrideH-config.PaddingH,
						)
					}

					if !useMax {
						AvgPool2DStride1RowFP16NEONAsm(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth,
							outRow*config.StrideH-config.PaddingH,
						)
					}
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					value := poolWindowTyped(
						input, channelOffsetIn,
						loadInput,
						inHeight, inWidth,
						outRow, outCol,
						config, useMax,
					)

					storeOutput(output, channelOffsetOut+outRow*outWidth+outCol, value)
				}
			}
		}
	}
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
