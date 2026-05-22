//go:build amd64

package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"golang.org/x/sys/cpu"
)

func Conv2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !cpu.X86.HasAVX512F {
		Conv2DTypedScalar(
			dtype.BFloat16,
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	if conv2DConfigNEONEligible(config) {
		conv2DBFloat16Stride1RowAVX512Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DBFloat16GeneralAVX512Native(
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
	if !cpu.X86.HasAVX512F {
		Conv2DTypedScalar(
			dtype.Float16,
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	if conv2DConfigNEONEligible(config) {
		conv2DFloat16Stride1RowAVX512Native(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DFloat16GeneralAVX512Native(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func conv2DBFloat16Stride1RowAVX512Native(
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
					Conv2dStride1RowBF16AVX512Asm(
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

func conv2DFloat16Stride1RowAVX512Native(
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
					Conv2dStride1RowFP16AVX512Asm(
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

func conv2DBFloat16GeneralAVX512Native(
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
		Conv2dPatchDotBF16AVX512Asm,
	)
}

func conv2DFloat16GeneralAVX512Native(
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
		Conv2dPatchDotFP16AVX512Asm,
	)
}

func Conv1DBFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	if !cpu.X86.HasAVX512F || !conv1DConfigNEONEligible(config) {
		Conv1DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength)

		return
	}

	loadInput, storeOutput := elementAccessors(dtype.BFloat16)
	loadWeight, _ := elementAccessors(dtype.BFloat16)
	loadBias, _ := elementAccessors(dtype.BFloat16)

	inWStride := inLength
	inCStride := inLength
	weightWStride := kernelLength
	weightCStride := kernelLength

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inLength

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelLength
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outLength
			biasValue := loadBias(bias, outChIndex)
			blockCols := outLength &^ 3

			if blockCols > 0 {
				Conv2dStride1RowBF16AVX512Asm(
					(*uint16)(unsafe.Add(output, uintptr(outputChannelOffset)*2)),
					(*uint16)(unsafe.Add(input, uintptr(inputBatchOffset)*2)),
					(*uint16)(unsafe.Add(weight, uintptr(weightChannelOffset)*2)),
					biasValue,
					blockCols,
					inChannels, 1, kernelLength,
					inWStride, inCStride,
					weightWStride, weightCStride,
					0, 0,
				)
			}

			for outIndex := blockCols; outIndex < outLength; outIndex++ {
				pixelValue := conv1DPixelTyped(
					config,
					input, weight,
					loadInput, loadWeight,
					inputBatchOffset, weightChannelOffset,
					inChannels, inLength, kernelLength,
					outIndex, biasValue,
				)

				storeOutput(output, outputChannelOffset+outIndex, pixelValue)
			}
		}
	}
}

func Conv1DFloat16Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	if !cpu.X86.HasAVX512F || !conv1DConfigNEONEligible(config) {
		Conv1DTypedScalar(dtype.Float16, config, input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength)

		return
	}

	loadInput, storeOutput := elementAccessors(dtype.Float16)
	loadWeight, _ := elementAccessors(dtype.Float16)
	loadBias, _ := elementAccessors(dtype.Float16)

	inWStride := inLength
	inCStride := inLength
	weightWStride := kernelLength
	weightCStride := kernelLength

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inLength

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelLength
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outLength
			biasValue := loadBias(bias, outChIndex)
			blockCols := outLength &^ 3

			if blockCols > 0 {
				Conv2dStride1RowFP16AVX512Asm(
					(*uint16)(unsafe.Add(output, uintptr(outputChannelOffset)*2)),
					(*uint16)(unsafe.Add(input, uintptr(inputBatchOffset)*2)),
					(*uint16)(unsafe.Add(weight, uintptr(weightChannelOffset)*2)),
					biasValue,
					blockCols,
					inChannels, 1, kernelLength,
					inWStride, inCStride,
					weightWStride, weightCStride,
					0, 0,
				)
			}

			for outIndex := blockCols; outIndex < outLength; outIndex++ {
				pixelValue := conv1DPixelTyped(
					config,
					input, weight,
					loadInput, loadWeight,
					inputBatchOffset, weightChannelOffset,
					inChannels, inLength, kernelLength,
					outIndex, biasValue,
				)

				storeOutput(output, outputChannelOffset+outIndex, pixelValue)
			}
		}
	}
}

func Conv3DBFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	if !cpu.X86.HasAVX512F {
		Conv3DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
			batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)

		return
	}

	conv3DReducedNative(
		config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		dtype.BFloat16,
		Conv2dPatchDotBF16AVX512Asm,
	)
}

func Conv3DFloat16Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	if !cpu.X86.HasAVX512F {
		Conv3DTypedScalar(dtype.Float16, config, input, weight, bias, output,
			batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW)

		return
	}

	conv3DReducedNative(
		config, input, weight, bias, output,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		dtype.Float16,
		Conv2dPatchDotFP16AVX512Asm,
	)
}

func ConvTranspose2DBFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !cpu.X86.HasAVX512F || !ConvTranspose2DConfigNEONEligible(config) {
		ConvTranspose2DTypedScalar(dtype.BFloat16, config, input, weight, bias, output,
			batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)

		return
	}

	convTranspose2DReducedEligibleNative(
		config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		dtype.BFloat16,
		ConvTranspose2dStride1RowBF16AVX512,
	)
}

func ConvTranspose2DFloat16Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !cpu.X86.HasAVX512F || !ConvTranspose2DConfigNEONEligible(config) {
		ConvTranspose2DTypedScalar(dtype.Float16, config, input, weight, bias, output,
			batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth)

		return
	}

	convTranspose2DReducedEligibleNative(
		config, input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		dtype.Float16,
		ConvTranspose2dStride1RowFP16AVX512,
	)
}
