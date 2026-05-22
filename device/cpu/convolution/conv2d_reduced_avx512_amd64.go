//go:build amd64

package convolution

//go:noescape
func Conv2dStride1RowBF16AVX512Asm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotBF16AVX512Asm(weight, patch *uint16, n int) float32

//go:noescape
func Conv2dStride1RowFP16AVX512Asm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride, wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotFP16AVX512Asm(weight, patch *uint16, n int) float32

//go:noescape
func ConvTranspose2dTapBF16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)

//go:noescape
func ConvTranspose2dTapFP16AVX512Asm(outRow *uint16, weightVal float32, inputCol *uint16, outCols int)
