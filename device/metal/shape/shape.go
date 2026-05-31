package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Shape implements device.Shape for the Metal backend.
*/
type Shape struct {
	host Host
}

/*
New wires a Shape receiver to its Metal dispatch host.
*/
func New(host Host) Shape {
	return Shape{host: host}
}

/*
Host is the Metal dispatch surface shape operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchCopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType)
	DispatchConcat(left, right, output unsafe.Pointer, format dtype.DType)
	DispatchGather(source, indices, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType)
	DispatchLastToken(input, output unsafe.Pointer, batch, seq, hidden int, format dtype.DType)
	DispatchMaskedFill(input, mask, fill, output unsafe.Pointer, count int, format dtype.DType)
	DispatchMergeHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType)
	DispatchPageGather(storage, pageTable, pageSize, output unsafe.Pointer, format dtype.DType)
	DispatchPageGatherWithLiveRows(
		storage, pageTable, pageSize, output unsafe.Pointer,
		liveRows int,
		format dtype.DType,
	)
	DispatchPageWrite(
		storage, values, pageIDs, offsets, output unsafe.Pointer,
		pageSize int,
		format dtype.DType,
	)
	DispatchReshape(input, output unsafe.Pointer, count int, format dtype.DType)
	DispatchScatter(target, indices, updates, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType)
	DispatchSlice(input, output unsafe.Pointer, dim, start, end int, format dtype.DType)
	DispatchSplit2(input, left, right unsafe.Pointer, format dtype.DType)
	DispatchSplitHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType)
	DispatchTranspose(input, permutation, output unsafe.Pointer, rank int, format dtype.DType)
	DispatchTranspose2D(input, output unsafe.Pointer, rows, cols int, format dtype.DType)
	DispatchUpsampleNearest2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchViewAsHeads(input, output unsafe.Pointer, batch, seq, numHeads, headDim int, format dtype.DType)
	DispatchWhere(mask, positive, negative, output unsafe.Pointer, count int, format dtype.DType)
}
