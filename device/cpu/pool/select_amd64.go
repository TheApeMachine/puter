//go:build amd64

package pool

import (
	"unsafe"

	"github.com/theapemachine/puter/device/cpu/reduction"
	"golang.org/x/sys/cpu"
)

func Pool2DFloat32Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth
	inputView := float32View(input, inputLength)
	outputView := float32View(output, outputLength)

	if !poolConfigNEONEligible(config) {
		Pool2DFloat32Scalar(
			config, inputView, outputView,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)

		return
	}

	if cpu.X86.HasAVX512F {
		pool2DFloat32FastRowNative(
			config, inputView, outputView,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax, poolF32AVX512Kernels,
		)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		pool2DFloat32FastRowNative(
			config, inputView, outputView,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax, poolF32AVX2Kernels,
		)

		return
	}

	if cpu.X86.HasSSE2 {
		pool2DFloat32FastRowNative(
			config, inputView, outputView,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax, poolF32SSE2Kernels,
		)

		return
	}

	Pool2DFloat32Scalar(
		config, inputView, outputView,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func pool2DFloat32FastRowAVX512Native(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	pool2DFloat32FastRowNative(
		config, inputView, outputView,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax, poolF32AVX512Kernels,
	)
}

func PoolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return reduction.ReduceMaxFloat32Native(scratch[:elementCount])
}

func PoolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return sumFloat32Sequential(scratch[:elementCount]) / float32(elementCount)
}

func AdaptivePool2DFloat32Native(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth
	inputView := float32View(input, inputLength)
	outputView := float32View(output, outputLength)

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffset := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channel := inputView[channelOffset : channelOffset+inHeight*inWidth]
			outputOffset := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				startRow := (outRow * inHeight) / outHeight
				endRow := ((outRow + 1) * inHeight) / outHeight

				for outCol := range outWidth {
					startCol := (outCol * inWidth) / outWidth
					endCol := ((outCol + 1) * inWidth) / outWidth
					outputIndex := outputOffset + outRow*outWidth + outCol

					if useMax {
						outputView[outputIndex] = PoolWindowMaxFloat32Native(
							channel, inWidth,
							startRow, endRow, startCol, endCol,
						)

						continue
					}

					outputView[outputIndex] = PoolWindowAvgFloat32Native(
						channel, inWidth,
						startRow, endRow, startCol, endCol,
					)
				}
			}
		}
	}
}
