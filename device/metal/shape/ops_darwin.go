//go:build darwin && cgo

package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (shape Shape) Concat(left, right, output unsafe.Pointer, format dtype.DType) {
	shape.host.DispatchConcat(left, right, output, format)
}

func (shape Shape) CopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType) {
	shape.host.DispatchCopyContiguous(dst, src, count, format)
}

func (shape Shape) Gather(source, indices, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType) {
	shape.host.DispatchGather(source, indices, output, outerDim, innerDim, format)
}

func (shape Shape) LastToken(input, output unsafe.Pointer, batch, seq, hidden int, format dtype.DType) {
	shape.host.DispatchLastToken(input, output, batch, seq, hidden, format)
}

func (shape Shape) MaskedFill(input, mask, fill, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.DispatchMaskedFill(input, mask, fill, output, count, format)
}

func (shape Shape) MergeHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType) {
	shape.host.DispatchMergeHeads(input, output, batch, seq, heads, headDim, format)
}

func (shape Shape) PageGather(storage, pageTable, pageSize, output unsafe.Pointer, format dtype.DType) {
	shape.host.DispatchPageGather(storage, pageTable, pageSize, output, format)
}

func (shape Shape) PageGatherWithLiveRows(
	storage, pageTable, pageSize, output unsafe.Pointer,
	liveRows int,
	format dtype.DType,
) {
	shape.host.DispatchPageGatherWithLiveRows(storage, pageTable, pageSize, output, liveRows, format)
}

func (shape Shape) PageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageSize int,
	format dtype.DType,
) {
	shape.host.DispatchPageWrite(storage, values, pageIDs, offsets, output, pageSize, format)
}

func (shape Shape) Reshape(input, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.DispatchReshape(input, output, count, format)
}

func (shape Shape) Scatter(target, indices, updates, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType) {
	shape.host.DispatchScatter(target, indices, updates, output, outerDim, innerDim, format)
}

func (shape Shape) Slice(input, output unsafe.Pointer, dim, start, end int, format dtype.DType) {
	shape.host.DispatchSlice(input, output, dim, start, end, format)
}

func (shape Shape) Split2(input, left, right unsafe.Pointer, format dtype.DType) {
	shape.host.DispatchSplit2(input, left, right, format)
}

func (shape Shape) SplitHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType) {
	shape.host.DispatchSplitHeads(input, output, batch, seq, heads, headDim, format)
}

func (shape Shape) Transpose(input, permutation, output unsafe.Pointer, rank int, format dtype.DType) {
	shape.host.DispatchTranspose(input, permutation, output, rank, format)
}

func (shape Shape) Transpose2D(input, output unsafe.Pointer, rows, cols int, format dtype.DType) {
	shape.host.DispatchTranspose2D(input, output, rows, cols, format)
}

func (shape Shape) UpsampleNearest2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	shape.host.DispatchUpsampleNearest2D(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format,
	)
}

func (shape Shape) ViewAsHeads(input, output unsafe.Pointer, batch, seq, numHeads, headDim int, format dtype.DType) {
	shape.host.DispatchViewAsHeads(input, output, batch, seq, numHeads, headDim, format)
}

func (shape Shape) Where(mask, positive, negative, output unsafe.Pointer, count int, format dtype.DType) {
	shape.host.DispatchWhere(mask, positive, negative, output, count, format)
}
