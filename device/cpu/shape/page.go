package shape

import "github.com/theapemachine/manifesto/tensor"

/*
runPageWrite writes logical rows into [pages, page_size, ...] storage.
Args: storage, values, page_ids, offsets, page_size, output.
*/
func runPageWrite(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	storageBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	valueBytes, err := aliasedBytes(args[1])

	if err != nil {
		return err
	}

	pageIDs, err := args[2].Int32Native()

	if err != nil {
		return err
	}

	offsets, err := args[3].Int32Native()

	if err != nil {
		return err
	}

	pageSize, err := int32ScalarTensor(args[4])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[5])

	if err != nil {
		return err
	}

	if len(storageBytes) != len(outBytes) {
		return tensor.ErrShapeMismatch
	}

	copyContiguousElements(outBytes, storageBytes, args[0].Len(), mustElementByteSize(args[0]))

	return writePageRows(args[0], args[1], pageIDs, offsets, int(pageSize), valueBytes, outBytes)
}

/*
runPageGather reads a logical sequence from [pages, page_size, ...] storage.
Args: storage, page_table, page_size, output.
*/
func runPageGather(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	storageBytes, err := aliasedBytes(args[0])

	if err != nil {
		return err
	}

	pageTable, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	pageSize, err := int32ScalarTensor(args[2])

	if err != nil {
		return err
	}

	outBytes, err := aliasedBytes(args[3])

	if err != nil {
		return err
	}

	return gatherPageRows(args[0], args[3], pageTable, int(pageSize), storageBytes, outBytes)
}

func writePageRows(
	storage tensor.Tensor,
	values tensor.Tensor,
	pageIDs []int32,
	offsets []int32,
	pageSize int,
	valueBytes []byte,
	outBytes []byte,
) error {
	storageDims := storage.Shape().Dims()
	valueDims := values.Shape().Dims()

	if len(storageDims) < 2 || len(valueDims) != len(storageDims)-1 {
		return tensor.ErrShapeMismatch
	}

	rowCount := valueDims[0]

	if rowCount != len(pageIDs) || rowCount != len(offsets) || storageDims[1] != pageSize {
		return tensor.ErrShapeMismatch
	}

	rowElements, err := trailingElementCount(valueDims[1:], storageDims[2:])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(storage)

	if err != nil {
		return err
	}

	rowBytes := rowElements * elementSize

	for rowIndex := range rowCount {
		pageID := int(pageIDs[rowIndex])
		pageOffset := int(offsets[rowIndex])

		if pageID < 0 || pageID >= storageDims[0] || pageOffset < 0 || pageOffset >= pageSize {
			return tensor.ErrShapeMismatch
		}

		storageRow := pageID*pageSize + pageOffset
		copyContiguousElements(
			outBytes[storageRow*rowBytes:(storageRow+1)*rowBytes],
			valueBytes[rowIndex*rowBytes:(rowIndex+1)*rowBytes],
			rowElements,
			elementSize,
		)
	}

	return nil
}

func gatherPageRows(
	storage tensor.Tensor,
	output tensor.Tensor,
	pageTable []int32,
	pageSize int,
	storageBytes []byte,
	outBytes []byte,
) error {
	storageDims := storage.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(storageDims) < 2 || len(outDims) != len(storageDims)-1 || storageDims[1] != pageSize {
		return tensor.ErrShapeMismatch
	}

	rowElements, err := trailingElementCount(outDims[1:], storageDims[2:])

	if err != nil {
		return err
	}

	elementSize, err := elementByteSize(storage)

	if err != nil {
		return err
	}

	rowBytes := rowElements * elementSize
	outRows := outDims[0]

	if len(outBytes) != outRows*rowBytes {
		return tensor.ErrShapeMismatch
	}

	for rowIndex := range outRows {
		tableIndex := rowIndex / pageSize

		if tableIndex >= len(pageTable) {
			return tensor.ErrShapeMismatch
		}

		pageID := int(pageTable[tableIndex])
		pageOffset := rowIndex % pageSize

		if pageID < 0 || pageID >= storageDims[0] {
			return tensor.ErrShapeMismatch
		}

		storageRow := pageID*pageSize + pageOffset
		copyContiguousElements(
			outBytes[rowIndex*rowBytes:(rowIndex+1)*rowBytes],
			storageBytes[storageRow*rowBytes:(storageRow+1)*rowBytes],
			rowElements,
			elementSize,
		)
	}

	return nil
}

func trailingElementCount(left []int, right []int) (int, error) {
	if len(left) != len(right) {
		return 0, tensor.ErrShapeMismatch
	}

	count := 1

	for index, dimension := range left {
		if dimension != right[index] {
			return 0, tensor.ErrShapeMismatch
		}

		count *= dimension
	}

	return count, nil
}

func mustElementByteSize(arg tensor.Tensor) int {
	elementSize, err := elementByteSize(arg)

	if err != nil {
		panic(err)
	}

	return elementSize
}
