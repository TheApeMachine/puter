package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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

func conv3DReducedNative(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
	patchDot reducedPatchDotFn,
) {
	loadInput, storeOutput := elementAccessors(format)
	loadBias, _ := elementAccessors(format)

	patchLength := inChannels * kD * kH * kW
	patchScratch := make([]uint16, patchLength)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inD * inH * inW

		for outChIndex := range outChannels {
			weightOffset := outChIndex * inChannels * kD * kH * kW
			biasValue := loadBias(bias, outChIndex)

			for outDIndex := range outD {
				for outHIndex := range outH {
					for outWIndex := range outW {
						conv3DPatchGatherReduced(
							config,
							input, inputBatchOffset,
							loadInput, patchScratch,
							inChannels, inD, inH, inW,
							kD, kH, kW,
							outDIndex, outHIndex, outWIndex,
							format,
						)

						dotValue := patchDot(
							(*uint16)(unsafe.Add(weight, uintptr(weightOffset)*2)),
							&patchScratch[0],
							patchLength,
						)

						outIndex := (((batchIndex*outChannels+outChIndex)*outD+outDIndex)*outH+outHIndex)*outW + outWIndex
						storeOutput(output, outIndex, biasValue+dotValue)
					}
				}
			}
		}
	}
}

func conv3DPatchGatherReduced(
	config Conv3DConfig,
	input unsafe.Pointer,
	inputBatchOffset int,
	loadInput elementLoad,
	patchScratch []uint16,
	inChannels, inD, inH, inW, kD, kH, kW, outDIndex, outHIndex, outWIndex int,
	format dtype.DType,
) {
	patchIndex := 0

	for inChIndex := range inChannels {
		for kDIndex := range kD {
			inDPos := outDIndex*config.StrideD + kDIndex*config.DilationD - config.PaddingD

			for kHIndex := range kH {
				inHPos := outHIndex*config.StrideH + kHIndex*config.DilationH - config.PaddingH

				for kWIndex := range kW {
					inWPos := outWIndex*config.StrideW + kWIndex*config.DilationW - config.PaddingW
					value := float32(0)

					if inDPos >= 0 && inDPos < inD &&
						inHPos >= 0 && inHPos < inH &&
						inWPos >= 0 && inWPos < inW {
						inputIndex := inputBatchOffset +
							((inChIndex*inD+inDPos)*inH+inHPos)*inW + inWPos
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
}

type convTransposeStride1RowFn func(
	outputRow, inputChannel, weightBlock []uint16,
	outCols, kernelHeight, kernelWidth, inHeight, inWidth int,
	outRowIndex, blockStartCol int,
)

func convTranspose2DReducedEligibleNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
	stride1Row convTransposeStride1RowFn,
) {
	loadInput, _ := elementAccessors(format)
	loadWeight, _ := elementAccessors(format)
	loadBias, _ := elementAccessors(format)
	loadOutput, storeOutput := elementAccessors(format)

	inputLength := batch * inChannels * inHeight * inWidth
	weightLength := inChannels * outChannels * kernelHeight * kernelWidth
	outputLength := batch * outChannels * outHeight * outWidth

	inputView := uint16View(input, inputLength)
	weightView := uint16View(weight, weightLength)
	outputView := uint16View(output, outputLength)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth
		outputBatchOffset := batchIndex * outChannels * outHeight * outWidth

		for outChIndex := range outChannels {
			outputChannelOffset := outputBatchOffset + outChIndex*outHeight*outWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				for outCol := range outWidth {
					storeOutput(output, outputChannelOffset+outRow*outWidth+outCol, biasValue)
				}
			}

			for inChIndex := range inChannels {
				inputChannelOffset := inputBatchOffset + inChIndex*inHeight*inWidth
				weightChannelOffset := inChIndex*outChannels*kernelHeight*kernelWidth + outChIndex*kernelHeight*kernelWidth

				for outRow := range outHeight {
					outputRowOffset := outputChannelOffset + outRow*outWidth
					outputRow := outputView[outputRowOffset : outputRowOffset+outWidth]
					scalarPrefix := kernelWidth - 1

					if scalarPrefix > outWidth {
						scalarPrefix = outWidth
					}

					if outRow < kernelHeight-1 {
						for outCol := range outWidth {
							pixelValue := loadOutput(output, outputRowOffset+outCol)
							pixelValue += convTranspose2DPixelTyped(
								config,
								input, weight,
								loadInput, loadWeight,
								inputChannelOffset, weightChannelOffset,
								inHeight, inWidth,
								kernelHeight, kernelWidth,
								outRow, outCol,
							)
							storeOutput(output, outputRowOffset+outCol, pixelValue)
						}

						continue
					}

					for outCol := 0; outCol < scalarPrefix; outCol++ {
						pixelValue := loadOutput(output, outputRowOffset+outCol)
						pixelValue += convTranspose2DPixelTyped(
							config,
							input, weight,
							loadInput, loadWeight,
							inputChannelOffset, weightChannelOffset,
							inHeight, inWidth,
							kernelHeight, kernelWidth,
							outRow, outCol,
						)
						storeOutput(output, outputRowOffset+outCol, pixelValue)
					}

					blockCols := (outWidth - scalarPrefix) &^ 3
					inputBlockCols := (inWidth - scalarPrefix) &^ 3

					if inputBlockCols < blockCols {
						blockCols = inputBlockCols
					}

					if blockCols > 0 {
						stride1Row(
							outputRow[scalarPrefix:],
							inputView[inputChannelOffset:],
							weightView[weightChannelOffset:],
							blockCols,
							kernelHeight, kernelWidth, inHeight, inWidth,
							outRow, scalarPrefix,
						)
					}

					for outCol := scalarPrefix + blockCols; outCol < outWidth; outCol++ {
						pixelValue := loadOutput(output, outputRowOffset+outCol)
						pixelValue += convTranspose2DPixelTyped(
							config,
							input, weight,
							loadInput, loadWeight,
							inputChannelOffset, weightChannelOffset,
							inHeight, inWidth,
							kernelHeight, kernelWidth,
							outRow, outCol,
						)
						storeOutput(output, outputRowOffset+outCol, pixelValue)
					}
				}
			}
		}
	}
}

func uint16View(pointer unsafe.Pointer, length int) []uint16 {
	if length == 0 {
		return nil
	}

	return unsafe.Slice((*uint16)(pointer), length)
}
