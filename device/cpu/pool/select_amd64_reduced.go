//go:build amd64

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
	if poolConfigNEONEligible(config) && reducedFloatSIMDAvailable() {
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
	if poolConfigNEONEligible(config) && reducedFloatSIMDAvailable() {
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
	strideTwo := config.StrideH == 2 && config.StrideW == 2

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				blockCols := outWidth &^ 3
				ihStart := outRow*config.StrideH - config.PaddingH

				if blockCols > 0 && strideTwo && useMax && maxPool2x2RowBF16 != nil {
					maxPool2x2RowBF16(
						(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && strideTwo && !useMax && avgPool2x2RowBF16 != nil {
					avgPool2x2RowBF16(
						(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && !strideTwo && maxPoolStride1RowBF16 != nil {
					if useMax {
						maxPoolStride1RowBF16(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}

					if !useMax {
						avgPoolStride1RowBF16(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth, ihStart,
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
	strideTwo := config.StrideH == 2 && config.StrideW == 2

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				blockCols := outWidth &^ 3
				ihStart := outRow*config.StrideH - config.PaddingH

				if blockCols > 0 && strideTwo && useMax && maxPool2x2RowFP16 != nil {
					maxPool2x2RowFP16(
						(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && strideTwo && !useMax && avgPool2x2RowFP16 != nil {
					avgPool2x2RowFP16(
						(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && !strideTwo && maxPoolStride1RowFP16 != nil {
					if useMax {
						maxPoolStride1RowFP16(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}

					if !useMax {
						avgPoolStride1RowFP16(
							(*uint16)(unsafe.Add(output, uintptr(channelOffsetOut+outRow*outWidth)*2)),
							(*uint16)(unsafe.Add(input, uintptr(channelOffsetIn)*2)),
							blockCols,
							config.KernelH, config.KernelW,
							inWidth, ihStart,
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
