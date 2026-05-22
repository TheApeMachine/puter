//go:build amd64

package convolution

import "golang.org/x/sys/cpu"

type convStride1RowBF16Fn func(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

type convStride1RowFP16Fn func(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

var (
	convStride1RowBF16 convStride1RowBF16Fn
	convPatchDotBF16   reducedPatchDotFn
	convStride1RowFP16 convStride1RowFP16Fn
	convPatchDotFP16   reducedPatchDotFn
)

func init() {
	if cpu.X86.HasAVX512F || (cpu.X86.HasAVX2 && cpu.X86.HasFMA) {
		convStride1RowBF16 = Conv2dStride1RowBF16AVX512Asm
		convPatchDotBF16 = Conv2dPatchDotBF16AVX512Asm
		convStride1RowFP16 = Conv2dStride1RowFP16AVX512Asm
		convPatchDotFP16 = Conv2dPatchDotFP16AVX512Asm

		return
	}

	if cpu.X86.HasSSE2 {
		convStride1RowBF16 = Conv2dStride1RowBF16SSE2Asm
		convPatchDotBF16 = Conv2dPatchDotBF16SSE2Asm
		convStride1RowFP16 = Conv2dStride1RowFP16SSE2Asm
		convPatchDotFP16 = Conv2dPatchDotFP16SSE2Asm
	}
}
