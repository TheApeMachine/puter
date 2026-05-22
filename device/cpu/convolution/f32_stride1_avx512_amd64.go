//go:build amd64

package convolution

//go:noescape
func Conv2dStride1RowF32AVX512Asm(
	outRow, input, weight *float32,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)
