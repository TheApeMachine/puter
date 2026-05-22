//go:build amd64

package pool

//go:noescape
func MaxPool2DStride1RowBF16SSE2Asm(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)

//go:noescape
func AvgPool2DStride1RowBF16SSE2Asm(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)

//go:noescape
func MaxPool2DStride1RowFP16SSE2Asm(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)

//go:noescape
func AvgPool2DStride1RowFP16SSE2Asm(
	outRow, input *uint16,
	outCols, kH, kW, inHStride, ihStart int,
)
