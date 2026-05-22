//go:build amd64

package pool

type f32PoolRowKernel struct {
	maxPoolStride1 func(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)
	avgPoolStride1 func(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)
	maxPool2x2     func(outRow, input *float32, outCols, inWidth, ihStart int)
	avgPool2x2     func(outRow, input *float32, outCols, inWidth, ihStart int)
}

func pool2DFloat32FastRowNative(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
	kernels f32PoolRowKernel,
) {
	strideTwo := config.StrideH == 2 && config.StrideW == 2

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth
			channel := inputView[channelOffsetIn : channelOffsetIn+inHeight*inWidth]
			outChannel := outputView[channelOffsetOut : channelOffsetOut+outHeight*outWidth]

			for outRow := range outHeight {
				outputRow := outChannel[outRow*outWidth : (outRow+1)*outWidth]
				blockCols := len(outputRow) &^ 3
				ihStart := outRow*config.StrideH - config.PaddingH

				if blockCols > 0 && strideTwo && useMax {
					kernels.maxPool2x2(
						&outputRow[0], &channel[0],
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && strideTwo && !useMax {
					kernels.avgPool2x2(
						&outputRow[0], &channel[0],
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && !strideTwo {
					if useMax {
						kernels.maxPoolStride1(
							&outputRow[0], &channel[0],
							blockCols, config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}

					if !useMax {
						kernels.avgPoolStride1(
							&outputRow[0], &channel[0],
							blockCols, config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					outputRow[outCol] = poolWindow(
						channel, inHeight, inWidth,
						outRow, outCol, config, useMax,
					)
				}
			}
		}
	}
}

var poolF32AVX512Kernels = f32PoolRowKernel{
	MaxPool2DStride1RowAVX512Asm,
	AvgPool2DStride1RowAVX512Asm,
	MaxPool2x2Stride2RowAVX512Asm,
	AvgPool2x2Stride2RowAVX512Asm,
}

var poolF32AVX2Kernels = f32PoolRowKernel{
	MaxPool2DStride1RowAVX2Asm,
	AvgPool2DStride1RowAVX2Asm,
	MaxPool2x2Stride2RowAVX2Asm,
	AvgPool2x2Stride2RowAVX2Asm,
}

var poolF32SSE2Kernels = f32PoolRowKernel{
	MaxPool2DStride1RowSSE2Asm,
	AvgPool2DStride1RowSSE2Asm,
	MaxPool2x2Stride2RowSSE2Asm,
	AvgPool2x2Stride2RowSSE2Asm,
}
