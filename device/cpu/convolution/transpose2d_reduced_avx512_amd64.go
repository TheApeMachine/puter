//go:build amd64

package convolution

import (
	"github.com/theapemachine/manifesto/dtype"
)

func ConvTranspose2dStride1RowBF16AVX512(
	outputRow, inputChannel, weightBlock []uint16,
	outCols, kernelHeight, kernelWidth, inHeight, inWidth int,
	outRowIndex, blockStartCol int,
) {
	blockCols := outCols &^ 3
	if blockCols == 0 {
		return
	}

	for kernelRow := range kernelHeight {
		inputRow := outRowIndex - kernelRow
		if inputRow < 0 || inputRow >= inHeight {
			continue
		}

		inputRowOffset := inputRow * inWidth
		weightRowOffset := kernelRow * kernelWidth

		for kernelCol := range kernelWidth {
			inputCol := blockStartCol - kernelCol
			if inputCol < 0 || inputCol+blockCols > inWidth {
				continue
			}

			weightBits := weightBlock[weightRowOffset+kernelCol]
			weightValue := loadBF16FromBits(weightBits)

			ConvTranspose2dTapBF16AVX512Asm(
				&outputRow[0],
				weightValue,
				&inputChannel[inputRowOffset+inputCol],
				blockCols,
			)
		}
	}
}

func ConvTranspose2dStride1RowFP16AVX512(
	outputRow, inputChannel, weightBlock []uint16,
	outCols, kernelHeight, kernelWidth, inHeight, inWidth int,
	outRowIndex, blockStartCol int,
) {
	blockCols := outCols &^ 3
	if blockCols == 0 {
		return
	}

	for kernelRow := range kernelHeight {
		inputRow := outRowIndex - kernelRow
		if inputRow < 0 || inputRow >= inHeight {
			continue
		}

		inputRowOffset := inputRow * inWidth
		weightRowOffset := kernelRow * kernelWidth

		for kernelCol := range kernelWidth {
			inputCol := blockStartCol - kernelCol
			if inputCol < 0 || inputCol+blockCols > inWidth {
				continue
			}

			weightBits := weightBlock[weightRowOffset+kernelCol]
			weightValue := loadF16FromBits(weightBits)

			ConvTranspose2dTapFP16AVX512Asm(
				&outputRow[0],
				weightValue,
				&inputChannel[inputRowOffset+inputCol],
				blockCols,
			)
		}
	}
}

func loadBF16FromBits(bits uint16) float32 {
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func loadF16FromBits(bits uint16) float32 {
	half := dtype.Frombits(bits)
	return half.Float32()
}
