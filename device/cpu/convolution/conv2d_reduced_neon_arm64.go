//go:build arm64

package convolution

//go:noescape
func Conv2dStride1RowBF16NEONAsm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride int,
	wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotBF16NEONAsm(weight, patch *uint16, n int) float32

//go:noescape
func Conv2dStride1RowFP16NEONAsm(
	outRow, input, weight *uint16,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride int,
	wHStride, wCStride int,
	ihStart, iwStart int,
)

//go:noescape
func Conv2dPatchDotFP16NEONAsm(weight, patch *uint16, n int) float32
