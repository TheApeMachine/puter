package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Conv2DTypedScalar(
	format dtype.DType,
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	loadInput, storeOutput := elementAccessors(format)
	loadWeight, _ := elementAccessors(format)
	loadBias, _ := elementAccessors(format)

	if conv2DConfigNEONEligible(config) {
		conv2DTypedStride1Scalar(
			config,
			input, weight, bias, output,
			loadInput, loadWeight, loadBias, storeOutput,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DTypedGeneralScalar(
		config,
		input, weight, bias, output,
		loadInput, loadWeight, loadBias, storeOutput,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func conv2DTypedStride1Scalar(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	loadInput, loadWeight, loadBias elementLoad,
	storeOutput elementStore,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				for outCol := range outWidth {
					outIndex := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
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

func conv2DTypedGeneralScalar(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	loadInput, loadWeight, loadBias elementLoad,
	storeOutput elementStore,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			biasValue := loadBias(bias, outChIndex)

			for outRow := range outHeight {
				for outCol := range outWidth {
					outIndex := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
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

func conv2DPixelTyped(
	config Conv2DConfig,
	input, weight unsafe.Pointer,
	loadInput, loadWeight elementLoad,
	inputBatchOffset, weightChannelOffset int,
	inChannels, inHeight, inWidth,
	kernelHeight, kernelWidth,
	outRow, outCol int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kRow := range kernelHeight {
			inRow := outRow*config.StrideH + kRow*config.DilationH - config.PaddingH

			if inRow < 0 || inRow >= inHeight {
				continue
			}

			for kCol := range kernelWidth {
				inCol := outCol*config.StrideW + kCol*config.DilationW - config.PaddingW

				if inCol < 0 || inCol >= inWidth {
					continue
				}

				inputIndex := inputBatchOffset + (inChIndex*inHeight+inRow)*inWidth + inCol
				weightIndex := weightChannelOffset + (inChIndex*kernelHeight+kRow)*kernelWidth + kCol

				sum += loadInput(input, inputIndex) * loadWeight(weight, weightIndex)
			}
		}
	}

	return sum
}

func conv1DPixelTyped(
	config Conv1DConfig,
	input, weight unsafe.Pointer,
	loadInput, loadWeight elementLoad,
	inputBatchOffset, weightChannelOffset int,
	inChannels, inLength, kernelLength, outIndex int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kernelIndex := range kernelLength {
			inPos := outIndex*config.Stride + kernelIndex*config.Dilation - config.Padding

			if inPos < 0 || inPos >= inLength {
				continue
			}

			inputIndex := inputBatchOffset + inChIndex*inLength + inPos
			weightIndex := weightChannelOffset + inChIndex*kernelLength + kernelIndex

			sum += loadInput(input, inputIndex) * loadWeight(weight, weightIndex)
		}
	}

	return sum
}

func convTranspose2DPixelTyped(
	config Conv2DConfig,
	input, weight unsafe.Pointer,
	loadInput, loadWeight elementLoad,
	inputChannelOffset, weightChannelOffset int,
	inHeight, inWidth, kernelHeight, kernelWidth,
	outRow, outCol int,
) float32 {
	var sum float32

	for kernelRow := range kernelHeight {
		inputRowNumerator := outRow + config.PaddingH - kernelRow*config.DilationH
		if inputRowNumerator%config.StrideH != 0 {
			continue
		}

		inputRow := inputRowNumerator / config.StrideH
		if inputRow < 0 || inputRow >= inHeight {
			continue
		}

		sum += convTranspose2DPixelRowTyped(
			config, input, weight,
			loadInput, loadWeight,
			inputChannelOffset, weightChannelOffset,
			inputRow, outCol, inWidth, kernelRow, kernelWidth,
		)
	}

	return sum
}

func convTranspose2DPixelRowTyped(
	config Conv2DConfig,
	input, weight unsafe.Pointer,
	loadInput, loadWeight elementLoad,
	inputChannelOffset, weightChannelOffset int,
	inputRow, outCol, inWidth, kernelRow, kernelWidth int,
) float32 {
	var sum float32

	for kernelCol := range kernelWidth {
		inputColNumerator := outCol + config.PaddingW - kernelCol*config.DilationW
		if inputColNumerator%config.StrideW != 0 {
			continue
		}

		inputCol := inputColNumerator / config.StrideW
		if inputCol < 0 || inputCol >= inWidth {
			continue
		}

		inputIndex := inputChannelOffset + inputRow*inWidth + inputCol
		weightIndex := weightChannelOffset + kernelRow*kernelWidth + kernelCol

		sum += loadInput(input, inputIndex) * loadWeight(weight, weightIndex)
	}

	return sum
}

func Conv1DTypedScalar(
	format dtype.DType,
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	loadInput, storeOutput := elementAccessors(format)
	loadWeight, _ := elementAccessors(format)
	loadBias, _ := elementAccessors(format)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inLength

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelLength
			biasValue := loadBias(bias, outChIndex)

			for outIndex := range outLength {
				sum := biasValue

				for inChIndex := range inChannels {
					for kernelIndex := range kernelLength {
						inPos := outIndex*config.Stride + kernelIndex*config.Dilation - config.Padding

						if inPos < 0 || inPos >= inLength {
							continue
						}

						inputIndex := inputBatchOffset + inChIndex*inLength + inPos
						weightIndex := weightChannelOffset + inChIndex*kernelLength + kernelIndex

						sum += loadInput(input, inputIndex) * loadWeight(weight, weightIndex)
					}
				}

				outputIndex := (batchIndex*outChannels+outChIndex)*outLength + outIndex
				storeOutput(output, outputIndex, sum)
			}
		}
	}
}

func Conv3DTypedScalar(
	format dtype.DType,
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	loadInput, storeOutput := elementAccessors(format)
	loadWeight, _ := elementAccessors(format)
	loadBias, _ := elementAccessors(format)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inD * inH * inW

		for outChIndex := range outChannels {
			weightOffset := outChIndex * inChannels * kD * kH * kW
			biasValue := loadBias(bias, outChIndex)

			for outDIndex := range outD {
				for outHIndex := range outH {
					for outWIndex := range outW {
						outputIndex := (((batchIndex*outChannels+outChIndex)*outD+outDIndex)*outH+outHIndex)*outW + outWIndex
						pixelValue := conv3DPixelTyped(
							config,
							input, weight,
							loadInput, loadWeight,
							inputBatchOffset, weightOffset,
							inChannels, inD, inH, inW,
							kD, kH, kW,
							outDIndex, outHIndex, outWIndex,
							biasValue,
						)

						storeOutput(output, outputIndex, pixelValue)
					}
				}
			}
		}
	}
}

func conv3DPixelTyped(
	config Conv3DConfig,
	input, weight unsafe.Pointer,
	loadInput, loadWeight elementLoad,
	inputBatchOffset, weightOffset int,
	inChannels, inD, inH, inW, kD, kH, kW, outDIndex, outHIndex, outWIndex int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kDIndex := range kD {
			for kHIndex := range kH {
				for kWIndex := range kW {
					inDPos := outDIndex*config.StrideD + kDIndex*config.DilationD - config.PaddingD
					inHPos := outHIndex*config.StrideH + kHIndex*config.DilationH - config.PaddingH
					inWPos := outWIndex*config.StrideW + kWIndex*config.DilationW - config.PaddingW

					if inDPos < 0 || inDPos >= inD ||
						inHPos < 0 || inHPos >= inH ||
						inWPos < 0 || inWPos >= inW {
						continue
					}

					inputIndex := inputBatchOffset +
						((inChIndex*inD+inDPos)*inH+inHPos)*inW + inWPos
					weightIndex := weightOffset +
						((inChIndex*kD+kDIndex)*kH+kHIndex)*kW + kWIndex

					sum += loadInput(input, inputIndex) * loadWeight(weight, weightIndex)
				}
			}
		}
	}

	return sum
}

func ConvTranspose2DTypedScalar(
	format dtype.DType,
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	loadInput, _ := elementAccessors(format)
	loadWeight, _ := elementAccessors(format)
	loadBias, _ := elementAccessors(format)
	loadOutput, storeOutput := elementAccessors(format)

	for batchIndex := range batch {
		for outChIndex := range outChannels {
			channelOffset := (batchIndex*outChannels + outChIndex) * outHeight * outWidth
			biasValue := loadBias(bias, outChIndex)

			for rowIndex := range outHeight {
				for colIndex := range outWidth {
					channelIndex := channelOffset + rowIndex*outWidth + colIndex
					storeOutput(output, channelIndex, biasValue)
				}
			}
		}
	}

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth
		outputBatchOffset := batchIndex * outChannels * outHeight * outWidth

		for inChIndex := range inChannels {
			inputChannelOffset := inputBatchOffset + inChIndex*inHeight*inWidth

			for outChIndex := range outChannels {
				weightChannelOffset := (inChIndex*outChannels + outChIndex) * kernelHeight * kernelWidth
				outputChannelOffset := outputBatchOffset + outChIndex*outHeight*outWidth

				for inRow := range inHeight {
					for inCol := range inWidth {
						inputValue := loadInput(input, inputChannelOffset+inRow*inWidth+inCol)

						convTranspose2DScatterTyped(
							config,
							weight, output,
							loadWeight, storeOutput, loadOutput,
							inputValue,
							weightChannelOffset, outputChannelOffset,
							inRow, inCol,
							kernelHeight, kernelWidth, outHeight, outWidth,
						)
					}
				}
			}
		}
	}
}

func convTranspose2DScatterTyped(
	config Conv2DConfig,
	weight, output unsafe.Pointer,
	loadWeight elementLoad,
	storeOutput elementStore,
	loadOutput elementLoad,
	inputValue float32,
	weightChannelOffset, outputChannelOffset, inRow, inCol,
	kernelHeight, kernelWidth, outHeight, outWidth int,
) {
	for kRow := range kernelHeight {
		outRow := inRow*config.StrideH + kRow*config.DilationH - config.PaddingH

		if outRow < 0 || outRow >= outHeight {
			continue
		}

		outColBase := inCol*config.StrideW - config.PaddingW

		for kCol := range kernelWidth {
			outCol := outColBase + kCol*config.DilationW

			if outCol < 0 || outCol >= outWidth {
				continue
			}

			weightIndex := weightChannelOffset + kRow*kernelWidth + kCol
			outIndex := outputChannelOffset + outRow*outWidth + outCol
			accumulated := loadOutput(output, outIndex) +
				inputValue*loadWeight(weight, weightIndex)

			storeOutput(output, outIndex, accumulated)
		}
	}
}
