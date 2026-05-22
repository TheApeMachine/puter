//go:build amd64

package pool

import "golang.org/x/sys/cpu"

type poolStride1RowBF16Fn func(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)

type poolStride2RowBF16Fn func(
	outRow, input *uint16,
	outCols, inWidth, ihStart int,
)

type poolStride1RowFP16Fn func(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)

type poolStride2RowFP16Fn func(
	outRow, input *uint16,
	outCols, inWidth, ihStart int,
)

var (
	maxPoolStride1RowBF16 poolStride1RowBF16Fn
	avgPoolStride1RowBF16 poolStride1RowBF16Fn
	maxPool2x2RowBF16     poolStride2RowBF16Fn
	avgPool2x2RowBF16     poolStride2RowBF16Fn

	maxPoolStride1RowFP16 poolStride1RowFP16Fn
	avgPoolStride1RowFP16 poolStride1RowFP16Fn
	maxPool2x2RowFP16     poolStride2RowFP16Fn
	avgPool2x2RowFP16     poolStride2RowFP16Fn
)

func init() {
	if cpu.X86.HasAVX512F || (cpu.X86.HasAVX2 && cpu.X86.HasFMA) {
		maxPoolStride1RowBF16 = MaxPool2DStride1RowBF16AVX512Asm
		avgPoolStride1RowBF16 = AvgPool2DStride1RowBF16AVX512Asm
		maxPool2x2RowBF16 = MaxPool2x2Stride2RowBF16AVX512Asm
		avgPool2x2RowBF16 = AvgPool2x2Stride2RowBF16AVX512Asm

		maxPoolStride1RowFP16 = MaxPool2DStride1RowFP16AVX512Asm
		avgPoolStride1RowFP16 = AvgPool2DStride1RowFP16AVX512Asm
		maxPool2x2RowFP16 = MaxPool2x2Stride2RowFP16AVX512Asm
		avgPool2x2RowFP16 = AvgPool2x2Stride2RowFP16AVX512Asm

		return
	}

	if cpu.X86.HasSSE2 {
		maxPoolStride1RowBF16 = MaxPool2DStride1RowBF16SSE2Asm
		avgPoolStride1RowBF16 = AvgPool2DStride1RowBF16SSE2Asm
		maxPool2x2RowBF16 = MaxPool2x2Stride2RowBF16SSE2Asm
		avgPool2x2RowBF16 = AvgPool2x2Stride2RowBF16SSE2Asm

		maxPoolStride1RowFP16 = MaxPool2DStride1RowFP16SSE2Asm
		avgPoolStride1RowFP16 = AvgPool2DStride1RowFP16SSE2Asm
		maxPool2x2RowFP16 = MaxPool2x2Stride2RowFP16SSE2Asm
		avgPool2x2RowFP16 = AvgPool2x2Stride2RowFP16SSE2Asm
	}
}
