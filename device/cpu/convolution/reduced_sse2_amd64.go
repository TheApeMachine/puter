//go:build amd64

package convolution

//go:noescape
func Conv2dStride1RowBF16SSE2Asm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotBF16SSE2Asm(weight, patch *uint16, n int) float32

//go:noescape
func Conv2dStride1RowFP16SSE2Asm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotFP16SSE2Asm(weight, patch *uint16, n int) float32
