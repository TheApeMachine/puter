package pool

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Pool2DTypedScalar(
	format dtype.DType,
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	loadInput, storeOutput := elementAccessors(format)

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				for outCol := range outWidth {
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

func poolWindowTyped(
	input unsafe.Pointer,
	channelOffset int,
	loadInput elementLoad,
	inHeight, inWidth int,
	outRow, outCol int,
	config PoolConfig,
	useMax bool,
) float32 {
	startRow := outRow*config.StrideH - config.PaddingH
	startCol := outCol*config.StrideW - config.PaddingW

	value := float32(math.Inf(-1))

	if !useMax {
		value = 0
	}

	count := 0

	for kernelRow := range config.KernelH {
		for kernelCol := range config.KernelW {
			row := startRow + kernelRow
			col := startCol + kernelCol

			if row < 0 || row >= inHeight || col < 0 || col >= inWidth {
				continue
			}

			candidate := loadInput(input, channelOffset+row*inWidth+col)
			count++

			switch {
			case useMax:
				if candidate > value {
					value = candidate
				}
			default:
				value += candidate
			}
		}
	}

	if !useMax && count > 0 {
		value /= float32(count)
	}

	return value
}

func AdaptivePool2DTypedScalar(
	format dtype.DType,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	loadInput, storeOutput := elementAccessors(format)

	for batchIndex := range batch {
		for channelIndex := range channels {
			for outRow := range outHeight {
				startRow := (outRow * inHeight) / outHeight
				endRow := ((outRow + 1) * inHeight) / outHeight

				for outCol := range outWidth {
					startCol := (outCol * inWidth) / outWidth
					endCol := ((outCol + 1) * inWidth) / outWidth

					value := adaptivePoolValueTyped(
						input, batchIndex, channelIndex, channels,
						inHeight, inWidth,
						loadInput,
						startRow, endRow, startCol, endCol,
						useMax,
					)

					outputIndex := ((batchIndex*channels+channelIndex)*outHeight+outRow)*outWidth + outCol
					storeOutput(output, outputIndex, value)
				}
			}
		}
	}
}

func adaptivePoolValueTyped(
	input unsafe.Pointer,
	batchIndex, channelIndex, channels, inHeight, inWidth int,
	loadInput elementLoad,
	startRow, endRow, startCol, endCol int,
	useMax bool,
) float32 {
	var sum float32
	maximum := float32(-1e30)
	count := 0

	for row := startRow; row < endRow; row++ {
		for col := startCol; col < endCol; col++ {
			index := ((batchIndex*channels+channelIndex)*inHeight+row)*inWidth + col
			value := loadInput(input, index)
			count++

			if useMax {
				if value > maximum {
					maximum = value
				}

				continue
			}

			sum += value
		}
	}

	if useMax {
		return maximum
	}

	if count == 0 {
		return 0
	}

	return sum / float32(count)
}
