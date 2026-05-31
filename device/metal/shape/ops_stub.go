//go:build !darwin || !cgo

package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (shape Shape) Concat(left, right, output unsafe.Pointer, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) CopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Gather(source, indices, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) LastToken(input, output unsafe.Pointer, batch, seq, hidden int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) MaskedFill(input, mask, fill, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) MergeHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) PageGather(storage, pageTable, pageSize, output unsafe.Pointer, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) PageGatherWithLiveRows(
	storage, pageTable, pageSize, output unsafe.Pointer,
	liveRows int,
	format dtype.DType,
) {
	shape.host.NeedsPlatform()
}

func (shape Shape) PageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageSize int,
	format dtype.DType,
) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Reshape(input, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Scatter(target, indices, updates, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Slice(input, output unsafe.Pointer, dim, start, end int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Split2(input, left, right unsafe.Pointer, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) SplitHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Transpose(input, permutation, output unsafe.Pointer, rank int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Transpose2D(input, output unsafe.Pointer, rows, cols int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) UpsampleNearest2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	shape.host.NeedsPlatform()
}

func (shape Shape) ViewAsHeads(input, output unsafe.Pointer, batch, seq, numHeads, headDim int, format dtype.DType) {
	shape.host.NeedsPlatform()
}

func (shape Shape) Where(mask, positive, negative, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.NeedsPlatform()
}
