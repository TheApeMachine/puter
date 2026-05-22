//go:build amd64

package convolution

import "golang.org/x/sys/cpu"

type convStride1RowF32Fn func(
	outRow, input, weight *float32,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

var convStride1RowF32 convStride1RowF32Fn

func init() {
	if cpu.X86.HasAVX512F {
		convStride1RowF32 = Conv2dStride1RowF32AVX512Asm

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		convStride1RowF32 = Conv2dStride1RowF32AVX2Asm

		return
	}

	if cpu.X86.HasSSE2 {
		convStride1RowF32 = Conv2dStride1RowF32SSE2Asm
	}
}
