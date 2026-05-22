//go:build arm64

package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Conv2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if conv2DConfigNEONEligible(config) {
		conv2DBFloat16Stride1RowNative(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DBFloat16GeneralNative(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func Conv2DFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if conv2DConfigNEONEligible(config) {
		conv2DFloat16Stride1RowNative(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DFloat16GeneralNative(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func conv2DBFloat16Stride1RowNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	loadInput, storeOutput := elementAccessors(dtype.BFloat16)
	loadWeight, _ := elementAccessors(dtype.BFloat16)
	loadBias, _ := elementAccessors(dtype.BFloat16)

	inHStride := inWidth
	inCStride := inHeight * inWidth
	weightHStride := kernelWidth
	weightCStride := kernelHeight * kernelWidth

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outHeight * outWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				blockCols := outWidth &^ 3

				if blockCols > 0 {
					Conv2dStride1RowBF16NEONAsm(
						(*uint16)(unsafe.Add(output, uintptr(outputChannelOffset+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(inputBatchOffset)*2)),
						(*uint16)(unsafe.Add(weight, uintptr(weightChannelOffset)*2)),
						biasValue,
						blockCols,
						inChannels, kernelHeight, kernelWidth,
						inHStride, inCStride,
						weightHStride, weightCStride,
						outRow, 0,
					)
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					outIndex := outputChannelOffset + outRow*outWidth + outCol
					pixelValue := conv2DPixelTyped(
						config,
						input, weight,
						loadInput, loadWeight,
						inputBatchOffset, weightChannelOffset,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
						biasValue,
					)

					storeOutput(output, outIndex, pixelValue)
				}
			}
		}
	}
}

func conv2DFloat16Stride1RowNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	loadInput, storeOutput := elementAccessors(dtype.Float16)
	loadWeight, _ := elementAccessors(dtype.Float16)
	loadBias, _ := elementAccessors(dtype.Float16)

	inHStride := inWidth
	inCStride := inHeight * inWidth
	weightHStride := kernelWidth
	weightCStride := kernelHeight * kernelWidth

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outHeight * outWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				blockCols := outWidth &^ 3

				if blockCols > 0 {
					Conv2dStride1RowFP16NEONAsm(
						(*uint16)(unsafe.Add(output, uintptr(outputChannelOffset+outRow*outWidth)*2)),
						(*uint16)(unsafe.Add(input, uintptr(inputBatchOffset)*2)),
						(*uint16)(unsafe.Add(weight, uintptr(weightChannelOffset)*2)),
						biasValue,
						blockCols,
						inChannels, kernelHeight, kernelWidth,
						inHStride, inCStride,
						weightHStride, weightCStride,
						outRow, 0,
					)
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					outIndex := outputChannelOffset + outRow*outWidth + outCol
					pixelValue := conv2DPixelTyped(
						config,
						input, weight,
						loadInput, loadWeight,
						inputBatchOffset, weightChannelOffset,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
						biasValue,
					)

					storeOutput(output, outIndex, pixelValue)
				}
			}
		}
	}
}

func conv2DBFloat16GeneralNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	conv2DReducedGeneralNative(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		dtype.BFloat16,
		Conv2dPatchDotBF16NEONAsm,
	)
}

func conv2DFloat16GeneralNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	conv2DReducedGeneralNative(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		dtype.Float16,
		Conv2dPatchDotFP16NEONAsm,
	)
}

type reducedPatchDotFn func(weight, patch *uint16, n int) float32

func conv2DReducedGeneralNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
	patchDot reducedPatchDotFn,
) {
	loadInput, storeOutput := elementAccessors(format)
	loadBias, _ := elementAccessors(format)

	patchLength := inChannels * kernelHeight * kernelWidth
	patchScratch := make([]uint16, patchLength)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				for outCol := range outWidth {
					conv2DPatchGatherReduced(
						config,
						input, inputBatchOffset,
						loadInput, patchScratch,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
						format,
					)

					dotValue := patchDot(
						(*uint16)(unsafe.Add(weight, uintptr(weightChannelOffset)*2)),
						&patchScratch[0],
						patchLength,
					)

					outIndex := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
					storeOutput(output, outIndex, biasValue+dotValue)
				}
			}
		}
	}
}

func conv2DPatchGatherReduced(
	config Conv2DConfig,
	input unsafe.Pointer,
	inputBatchOffset int,
	loadInput elementLoad,
	patchScratch []uint16,
	inChannels, inHeight, inWidth, kernelHeight, kernelWidth, outRow, outCol int,
	format dtype.DType,
) {
	patchIndex := 0

	for inChIndex := range inChannels {
		for kRow := range kernelHeight {
			inRow := outRow*config.StrideH + kRow*config.DilationH - config.PaddingH

			for kCol := range kernelWidth {
				inCol := outCol*config.StrideW + kCol*config.DilationW - config.PaddingW
				value := float32(0)

				if inRow >= 0 && inRow < inHeight && inCol >= 0 && inCol < inWidth {
					inputIndex := inputBatchOffset + (inChIndex*inHeight+inRow)*inWidth + inCol
					value = loadInput(input, inputIndex)
				}

				switch format {
				case dtype.Float16:
					patchScratch[patchIndex] = uint16(dtype.Fromfloat32(value).Bits())
				default:
					patchScratch[patchIndex] = uint16(dtype.NewBfloat16FromFloat32(value))
				}

				patchIndex++
			}
		}
	}
}

func Conv1DBFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	Conv1DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength)
}

func Conv1DFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	Conv1DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength)
}

func Conv3DBFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	Conv3DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)
}

func Conv3DFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	Conv3DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)
}

func ConvTranspose2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	ConvTranspose2DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)
}

func ConvTranspose2DFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	ConvTranspose2DTypedScalar(dtype.Float16, config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)
}
