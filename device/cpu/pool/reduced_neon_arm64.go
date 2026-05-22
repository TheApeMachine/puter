//go:build arm64

package pool

//go:noescape
func MaxPool2DStride1RowBF16NEONAsm(outRow, input *uint16, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func AvgPool2DStride1RowBF16NEONAsm(outRow, input *uint16, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func MaxPool2DStride1RowFP16NEONAsm(outRow, input *uint16, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func AvgPool2DStride1RowFP16NEONAsm(outRow, input *uint16, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func MaxPool2x2Stride2RowBF16NEONAsm(outRow, input *uint16, outCols, inWidth, ihStart int)

//go:noescape
func AvgPool2x2Stride2RowBF16NEONAsm(outRow, input *uint16, outCols, inWidth, ihStart int)

//go:noescape
func MaxPool2x2Stride2RowFP16NEONAsm(outRow, input *uint16, outCols, inWidth, ihStart int)

//go:noescape
func AvgPool2x2Stride2RowFP16NEONAsm(outRow, input *uint16, outCols, inWidth, ihStart int)
