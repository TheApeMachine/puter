package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/dispatch"
)

func requireShapeDType(format dtype.DType) int {
	elementSize := dispatch.ElementByteSize(format)

	if elementSize == 0 {
		panic("shape: unsupported dtype")
	}

	return elementSize
}

func (shape Shape) CopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementSize := requireShapeDType(format)

	if count == 0 {
		return
	}

	dstData, _, _, _ := dispatch.ResolvePointer(dst)
	srcData, _, _, _ := dispatch.ResolvePointer(src)

	if elementSize == 4 {
		copyContiguousF32Kernel(
			(*float32)(dstData),
			(*float32)(srcData),
			count,
		)

		return
	}

	byteCount := count * elementSize
	dstBytes := unsafe.Slice((*byte)(dstData), byteCount)
	srcBytes := unsafe.Slice((*byte)(srcData), byteCount)
	copy(dstBytes, srcBytes)
}

func (shape Shape) Where(
	mask, positive, negative, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireShapeDType(format)

	if count == 0 {
		return
	}

	if format != dtype.Float32 {
		panic("shape: Where requires float32")
	}

	maskBytes := unsafe.Slice((*byte)(mask), (count+7)/8)
	positiveData, _, _, _ := dispatch.ResolvePointer(positive)
	negativeData, _, _, _ := dispatch.ResolvePointer(negative)
	outputData, _, _, _ := dispatch.ResolvePointer(output)

	whereF32Kernel(
		(*float32)(outputData),
		(*float32)(positiveData),
		(*float32)(negativeData),
		maskBytes,
		count,
	)
}

func (shape Shape) MaskedFill(
	input, mask, fill, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	requireShapeDType(format)

	if count == 0 {
		return
	}

	if format != dtype.Float32 {
		panic("shape: MaskedFill requires float32")
	}

	maskBytes := unsafe.Slice((*byte)(mask), (count+7)/8)
	inputData, _, _, _ := dispatch.ResolvePointer(input)
	fillData, _, _, _ := dispatch.ResolvePointer(fill)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	fillValue := *(*float32)(fillData)

	maskedFillF32Kernel(
		(*float32)(outputData),
		(*float32)(inputData),
		fillValue,
		maskBytes,
		count,
	)
}

func (shape Shape) Reshape(input, output unsafe.Pointer, count int, format dtype.DType) {
	shape.CopyContiguous(output, input, count, format)
}

func (shape Shape) Transpose2D(
	input, output unsafe.Pointer,
	rows, cols int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if rows == 0 || cols == 0 {
		return
	}

	inputData, _, _, _ := dispatch.ResolvePointer(input)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	inputBytes := unsafe.Slice((*byte)(inputData), rows*cols*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), rows*cols*elementSize)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			copyElementAt(outputBytes, inputBytes, col*rows+row, row*cols+col, elementSize)
		}
	}
}

func (shape Shape) LastToken(
	input, output unsafe.Pointer,
	batch, seq, hidden int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if batch == 0 || seq == 0 || hidden == 0 {
		return
	}

	inputData, _, _, _ := dispatch.ResolvePointer(input)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	hiddenBytes := hidden * elementSize
	inputBytes := unsafe.Slice((*byte)(inputData), batch*seq*hidden*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), batch*hidden*elementSize)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		lastTokenOffset := ((batchIndex*seq + seq - 1) * hidden) * elementSize
		outOffset := batchIndex * hiddenBytes

		copyContiguousElements(
			outputBytes[outOffset:outOffset+hiddenBytes],
			inputBytes[lastTokenOffset:lastTokenOffset+hiddenBytes],
			hidden,
			elementSize,
		)
	}
}

func (shape Shape) MergeHeads(
	input, output unsafe.Pointer,
	batch, seq, heads, headDim int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if batch == 0 || seq == 0 || heads == 0 || headDim == 0 {
		return
	}

	inputData, _, _, _ := dispatch.ResolvePointer(input)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	headDimBytes := headDim * elementSize
	inputBytes := unsafe.Slice((*byte)(inputData), batch*seq*heads*headDim*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), batch*seq*heads*headDim*elementSize)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for seqIndex := 0; seqIndex < seq; seqIndex++ {
			for headIndex := 0; headIndex < heads; headIndex++ {
				inOffset := (((batchIndex*seq+seqIndex)*heads + headIndex) * headDim) * elementSize
				outOffset := (((batchIndex*seq + seqIndex) * heads * headDim) + headIndex*headDim) * elementSize

				copyContiguousElements(
					outputBytes[outOffset:outOffset+headDimBytes],
					inputBytes[inOffset:inOffset+headDimBytes],
					headDim,
					elementSize,
				)
			}
		}
	}
}

func (shape Shape) SplitHeads(
	input, output unsafe.Pointer,
	batch, seq, heads, headDim int,
	format dtype.DType,
) {
	shape.MergeHeads(input, output, batch, seq, heads, headDim, format)
}

func (shape Shape) ViewAsHeads(
	input, output unsafe.Pointer,
	batch, seq, numHeads, headDim int,
	format dtype.DType,
) {
	shape.CopyContiguous(output, input, batch*seq*numHeads*headDim, format)
}

func (shape Shape) UpsampleNearest2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 {
		return
	}

	inputData, _, _, _ := dispatch.ResolvePointer(input)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	inputBytes := unsafe.Slice((*byte)(inputData), batch*channels*inHeight*inWidth*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), batch*channels*outHeight*outWidth*elementSize)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			for outRow := 0; outRow < outHeight; outRow++ {
				inRow := outRow * inHeight / outHeight

				for outCol := 0; outCol < outWidth; outCol++ {
					inCol := outCol * inWidth / outWidth
					inIndex := ((batchIndex*channels+channelIndex)*inHeight+inRow)*inWidth + inCol
					outIndex := ((batchIndex*channels+channelIndex)*outHeight+outRow)*outWidth + outCol
					copyElementAt(outputBytes, inputBytes, outIndex, inIndex, elementSize)
				}
			}
		}
	}
}

func (shape Shape) Gather(
	source, indices, output unsafe.Pointer,
	outerDim, innerDim int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if outerDim == 0 || innerDim == 0 {
		return
	}

	_, outputCount, _, outputWrapped := dispatch.ResolvePointer(output)

	if !outputWrapped || outputCount%innerDim != 0 {
		panic("shape: gather requires dispatch.View on output")
	}

	outRows := outputCount / innerDim
	sourceData, _, _, _ := dispatch.ResolvePointer(source)
	indicesData, _, _, _ := dispatch.ResolvePointer(indices)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	sourceBytes := unsafe.Slice((*byte)(sourceData), outerDim*innerDim*elementSize)
	indicesNative := dispatch.Int32Slice(indicesData, outRows)
	outputBytes := unsafe.Slice((*byte)(outputData), outRows*innerDim*elementSize)
	innerBytes := innerDim * elementSize

	for resultIndex, sourceRow := range indicesNative {
		if int(sourceRow) < 0 || int(sourceRow) >= outerDim {
			panic("shape: gather index out of range")
		}

		copyContiguousElements(
			outputBytes[resultIndex*innerBytes:(resultIndex+1)*innerBytes],
			sourceBytes[int(sourceRow)*innerBytes:(int(sourceRow)+1)*innerBytes],
			innerDim,
			elementSize,
		)
	}
}

func (shape Shape) Scatter(
	target, indices, updates, output unsafe.Pointer,
	outerDim, innerDim int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if outerDim == 0 || innerDim == 0 {
		return
	}

	_, updateCount, _, updatesWrapped := dispatch.ResolvePointer(updates)

	if !updatesWrapped || updateCount%innerDim != 0 {
		panic("shape: scatter requires dispatch.View on updates")
	}

	updateRows := updateCount / innerDim
	targetData, _, _, _ := dispatch.ResolvePointer(target)
	indicesData, _, _, _ := dispatch.ResolvePointer(indices)
	updatesData, _, _, _ := dispatch.ResolvePointer(updates)
	outputData, _, _, _ := dispatch.ResolvePointer(output)
	targetBytes := unsafe.Slice((*byte)(targetData), outerDim*innerDim*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), len(targetBytes))
	updatesBytes := unsafe.Slice((*byte)(updatesData), updateRows*innerDim*elementSize)
	indicesNative := dispatch.Int32Slice(indicesData, updateRows)
	innerBytes := innerDim * elementSize

	copyContiguousElements(outputBytes, targetBytes, outerDim*innerDim, elementSize)

	for updateIndex, targetRow := range indicesNative {
		if int(targetRow) < 0 || int(targetRow) >= outerDim {
			panic("shape: scatter index out of range")
		}

		copyContiguousElements(
			outputBytes[int(targetRow)*innerBytes:(int(targetRow)+1)*innerBytes],
			updatesBytes[updateIndex*innerBytes:(updateIndex+1)*innerBytes],
			innerDim,
			elementSize,
		)
	}
}

func (shape Shape) Concat(left, right, output unsafe.Pointer, format dtype.DType) {
	elementSize := requireShapeDType(format)

	leftData, leftCount, _, leftWrapped := dispatch.ResolvePointer(left)
	rightData, rightCount, _, rightWrapped := dispatch.ResolvePointer(right)
	outputData, outputCount, _, outputWrapped := dispatch.ResolvePointer(output)

	if !leftWrapped || !rightWrapped || !outputWrapped {
		panic("shape: concat requires dispatch.View on all tensors")
	}

	if outputCount != leftCount+rightCount {
		panic("shape: concat output element count mismatch")
	}

	leftBytes := leftCount * elementSize
	rightBytes := rightCount * elementSize
	totalBytes := outputCount * elementSize

	leftSlice := unsafe.Slice((*byte)(leftData), leftBytes)
	rightSlice := unsafe.Slice((*byte)(rightData), rightBytes)
	outputSlice := unsafe.Slice((*byte)(outputData), totalBytes)

	copy(outputSlice[:leftBytes], leftSlice)
	copy(outputSlice[leftBytes:], rightSlice)
}

func (shape Shape) Split2(input, left, right unsafe.Pointer, format dtype.DType) {
	elementSize := requireShapeDType(format)

	inputData, inputCount, _, inputWrapped := dispatch.ResolvePointer(input)
	leftData, leftCount, _, leftWrapped := dispatch.ResolvePointer(left)
	rightData, rightCount, _, rightWrapped := dispatch.ResolvePointer(right)

	if !inputWrapped || !leftWrapped || !rightWrapped {
		panic("shape: split2 requires dispatch.View on all tensors")
	}

	if leftCount+rightCount != inputCount {
		panic("shape: split2 element count mismatch")
	}

	inputBytes := inputCount * elementSize
	leftBytes := leftCount * elementSize
	rightBytes := rightCount * elementSize

	inputSlice := unsafe.Slice((*byte)(inputData), inputBytes)
	leftSlice := unsafe.Slice((*byte)(leftData), leftBytes)
	rightSlice := unsafe.Slice((*byte)(rightData), rightBytes)

	copy(leftSlice, inputSlice[:len(leftSlice)])
	copy(rightSlice, inputSlice[len(leftSlice):])
}

func (shape Shape) Transpose(
	input, permutation, output unsafe.Pointer,
	rank int,
	format dtype.DType,
) {
	elementSize := requireShapeDType(format)

	if rank <= 0 {
		return
	}

	inputData, inputCount, inputDims, inputWrapped := dispatch.ResolvePointer(input)
	outputData, outputCount, outputDims, outputWrapped := dispatch.ResolvePointer(output)
	permutationData, permCount, _, permWrapped := dispatch.ResolvePointer(permutation)

	if !inputWrapped || !outputWrapped || !permWrapped || permCount != rank {
		panic("shape: transpose requires dispatch.View metadata")
	}

	if inputCount != outputCount || len(inputDims) != rank || len(outputDims) != rank {
		panic("shape: transpose shape mismatch")
	}

	permutationNative := dispatch.Int32Slice(permutationData, rank)
	inputBytes := unsafe.Slice((*byte)(inputData), inputCount*elementSize)
	outputBytes := unsafe.Slice((*byte)(outputData), outputCount*elementSize)
	inStrides := computeRowMajorStrides(inputDims)
	outStrides := computeRowMajorStrides(outputDims)

	for flatIndex := 0; flatIndex < inputCount; flatIndex++ {
		inCoords := flatToCoords(flatIndex, inputDims, inStrides)
		outCoords := make([]int, rank)

		for outAxis, inAxis := range permutationNative {
			outCoords[outAxis] = inCoords[inAxis]
		}

		outFlat := coordsToFlat(outCoords, outStrides)
		copyElementAt(outputBytes, inputBytes, outFlat, flatIndex, elementSize)
	}
}

func (shape Shape) Slice(
	input, output unsafe.Pointer,
	dim, start, end int,
	format dtype.DType,
) {
	panic("shape: Slice requires tensor metadata via execution intrinsics")
}

func (shape Shape) PageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageSize int,
	format dtype.DType,
) {
	if pageSize <= 0 {
		panic("shape: page write page_size must be positive")
	}

	storageData, _, storageDims, storageWrapped := dispatch.ResolvePointer(storage)
	valuesData, _, valueDims, valuesWrapped := dispatch.ResolvePointer(values)
	pageIDsData, pageIDCount, _, pageIDsWrapped := dispatch.ResolvePointer(pageIDs)
	offsetsData, offsetCount, _, offsetsWrapped := dispatch.ResolvePointer(offsets)
	outputData, _, _, outputWrapped := dispatch.ResolvePointer(output)

	if !storageWrapped || !valuesWrapped || !pageIDsWrapped || !offsetsWrapped || !outputWrapped {
		panic("shape: page write requires dispatch.View on all tensors")
	}

	if pageIDCount != offsetCount {
		panic("shape: page write ids/offsets length mismatch")
	}

	config, err := pageWriteConfigFromDims(storageDims, valueDims, pageSize)

	if err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32:
		pageWriteF32Kernel.kernel(
			(*float32)(storageData),
			(*float32)(valuesData),
			(*int32)(pageIDsData),
			(*int32)(offsetsData),
			(*float32)(outputData),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.valueRows,
		)
	case dtype.Float16, dtype.BFloat16:
		pageWriteU16Kernel.kernel(
			(*uint16)(storageData),
			(*uint16)(valuesData),
			(*int32)(pageIDsData),
			(*int32)(offsetsData),
			(*uint16)(outputData),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.valueRows,
		)
	default:
		panic("shape: page write unsupported dtype")
	}
}

func (shape Shape) PageGather(
	storage, pageTable, pageSize, output unsafe.Pointer,
	format dtype.DType,
) {
	shape.PageGatherWithLiveRows(storage, pageTable, pageSize, output, 0, format)
}

func (shape Shape) PageGatherWithLiveRows(
	storage, pageTable, pageSize, output unsafe.Pointer,
	liveRows int,
	format dtype.DType,
) {
	pageSizeValue := int(dispatch.Int32Scalar(pageSize))

	if pageSizeValue <= 0 {
		panic("shape: page gather page_size must be positive")
	}

	storageData, _, storageDims, storageWrapped := dispatch.ResolvePointer(storage)
	pageTableData, pageTableCount, _, pageTableWrapped := dispatch.ResolvePointer(pageTable)
	outputData, _, outputDims, outputWrapped := dispatch.ResolvePointer(output)

	if !storageWrapped || !pageTableWrapped || !outputWrapped {
		panic("shape: page gather requires dispatch.View on storage, page table, and output")
	}

	config, err := pageGatherConfigFromDims(storageDims, outputDims, pageTableCount, pageSizeValue, liveRows)

	if err != nil {
		panic(err)
	}

	switch format {
	case dtype.Float32:
		pageGatherF32Kernel.kernel(
			(*float32)(storageData),
			(*int32)(pageTableData),
			(*float32)(outputData),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.outRows,
		)
	case dtype.Float16, dtype.BFloat16:
		pageGatherU16Kernel.kernel(
			(*uint16)(storageData),
			(*int32)(pageTableData),
			(*uint16)(outputData),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.outRows,
		)
	default:
		panic("shape: page gather unsupported dtype")
	}
}

func pageWriteConfigFromDims(
	storageDims, valueDims []int,
	pageSize int,
) (pageWriteKernelConfig, error) {
	if len(storageDims) < 2 ||
		len(valueDims) != len(storageDims)-1 ||
		storageDims[1] != pageSize {
		return pageWriteKernelConfig{}, tensor.ErrShapeMismatch
	}

	inner := 1

	for index := 2; index < len(storageDims); index++ {
		inner *= storageDims[index]
	}

	return pageWriteKernelConfig{
		pageCount: storageDims[0],
		pageSize:  pageSize,
		inner:     inner,
		valueRows: valueDims[0],
	}, nil
}

func pageGatherConfigFromDims(
	storageDims, outputDims []int,
	pageTableCount, pageSize, liveRows int,
) (pageGatherKernelConfig, error) {
	if len(storageDims) < 2 ||
		len(outputDims) != len(storageDims)-1 ||
		storageDims[1] != pageSize {
		return pageGatherKernelConfig{}, tensor.ErrShapeMismatch
	}

	inner := 1

	for index := 2; index < len(storageDims); index++ {
		inner *= storageDims[index]
	}

	outRows := outputDims[0]

	if liveRows > 0 && liveRows < outRows {
		outRows = liveRows
	}

	maxRows := pageTableCount * pageSize

	if outRows > maxRows {
		return pageGatherKernelConfig{}, tensor.ErrShapeMismatch
	}

	return pageGatherKernelConfig{
		pageCount: storageDims[0],
		pageSize:  pageSize,
		inner:     inner,
		outRows:   outRows,
	}, nil
}
