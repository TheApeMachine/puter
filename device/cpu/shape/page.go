package shape

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
runPageWrite writes logical rows into [pages, page_size, ...] storage.
Args: storage, values, page_ids, offsets, page_size, output.
*/
func runPageWrite(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
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

	config, err := pageWriteConfig(args[0], args[1], args[5], pageIDs, offsets, int(pageSize))

	if err != nil {
		return err
	}

	return runTypedPageWrite(args[0], args[1], args[5], pageIDs, offsets, config)
}

/*
runPageGather reads a logical sequence from [pages, page_size, ...] storage.
Args: storage, page_table, page_size, output.
*/
func runPageGather(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	pageTable, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	pageSize, err := int32ScalarTensor(args[2])

	if err != nil {
		return err
	}

	config, err := pageGatherConfig(args[0], args[3], pageTable, int(pageSize))

	if err != nil {
		return err
	}

	return runTypedPageGather(args[0], args[3], pageTable, config)
}

type pageWriteKernelConfig struct {
	pageCount int
	pageSize  int
	inner     int
	valueRows int
}

type pageGatherKernelConfig struct {
	pageCount int
	pageSize  int
	inner     int
	outRows   int
}

func pageWriteConfig(
	storage tensor.Tensor,
	values tensor.Tensor,
	out tensor.Tensor,
	pageIDs []int32,
	offsets []int32,
	pageSize int,
) (pageWriteKernelConfig, error) {
	storageDims := storage.Shape().Dims()
	valueDims := values.Shape().Dims()

	if len(storageDims) < 2 ||
		len(valueDims) != len(storageDims)-1 ||
		!storage.Shape().Equal(out.Shape()) ||
		storage.DType() != values.DType() ||
		storage.DType() != out.DType() {
		return pageWriteKernelConfig{}, tensor.ErrShapeMismatch
	}

	rowCount := valueDims[0]

	if rowCount != len(pageIDs) || rowCount != len(offsets) || storageDims[1] != pageSize {
		return pageWriteKernelConfig{}, tensor.ErrShapeMismatch
	}

	rowElements, err := trailingElementCount(valueDims[1:], storageDims[2:])

	if err != nil {
		return pageWriteKernelConfig{}, err
	}

	for rowIndex := range rowCount {
		pageID := int(pageIDs[rowIndex])
		pageOffset := int(offsets[rowIndex])

		if pageID < 0 || pageID >= storageDims[0] || pageOffset < 0 || pageOffset >= pageSize {
			return pageWriteKernelConfig{}, tensor.ErrShapeMismatch
		}
	}

	return pageWriteKernelConfig{
		pageCount: storageDims[0],
		pageSize:  pageSize,
		inner:     rowElements,
		valueRows: rowCount,
	}, nil
}

func pageGatherConfig(
	storage tensor.Tensor,
	output tensor.Tensor,
	pageTable []int32,
	pageSize int,
) (pageGatherKernelConfig, error) {
	storageDims := storage.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(storageDims) < 2 ||
		len(outDims) != len(storageDims)-1 ||
		storageDims[1] != pageSize ||
		storage.DType() != output.DType() {
		return pageGatherKernelConfig{}, tensor.ErrShapeMismatch
	}

	rowElements, err := trailingElementCount(outDims[1:], storageDims[2:])

	if err != nil {
		return pageGatherKernelConfig{}, err
	}

	outRows := outDims[0]

	for rowIndex := range outRows {
		tableIndex := rowIndex / pageSize

		if tableIndex >= len(pageTable) {
			return pageGatherKernelConfig{}, tensor.ErrShapeMismatch
		}

		pageID := int(pageTable[tableIndex])

		if pageID < 0 || pageID >= storageDims[0] {
			return pageGatherKernelConfig{}, tensor.ErrShapeMismatch
		}
	}

	return pageGatherKernelConfig{
		pageCount: storageDims[0],
		pageSize:  pageSize,
		inner:     rowElements,
		outRows:   outRows,
	}, nil
}

func runTypedPageWrite(
	storage tensor.Tensor,
	values tensor.Tensor,
	out tensor.Tensor,
	pageIDs []int32,
	offsets []int32,
	config pageWriteKernelConfig,
) error {
	switch storage.DType() {
	case dtype.Float32:
		storageView, err := storage.Float32Native()
		if err != nil {
			return err
		}

		valueView, err := values.Float32Native()
		if err != nil {
			return err
		}

		outView, err := out.Float32Native()
		if err != nil {
			return err
		}

		pageWriteF32Kernel.kernel(
			&storageView[0], &valueView[0], &pageIDs[0], &offsets[0], &outView[0],
			config.pageCount, config.pageSize, config.inner, config.valueRows,
		)

		return nil
	case dtype.Float16:
		storageView, err := storage.Float16Native()
		if err != nil {
			return err
		}

		valueView, err := values.Float16Native()
		if err != nil {
			return err
		}

		outView, err := out.Float16Native()
		if err != nil {
			return err
		}

		pageWriteU16Kernel.kernel(
			(*uint16)(unsafe.Pointer(&storageView[0])),
			(*uint16)(unsafe.Pointer(&valueView[0])),
			&pageIDs[0],
			&offsets[0],
			(*uint16)(unsafe.Pointer(&outView[0])),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.valueRows,
		)

		return nil
	case dtype.BFloat16:
		storageView, err := storage.BFloat16Native()
		if err != nil {
			return err
		}

		valueView, err := values.BFloat16Native()
		if err != nil {
			return err
		}

		outView, err := out.BFloat16Native()
		if err != nil {
			return err
		}

		pageWriteU16Kernel.kernel(
			(*uint16)(unsafe.Pointer(&storageView[0])),
			(*uint16)(unsafe.Pointer(&valueView[0])),
			&pageIDs[0],
			&offsets[0],
			(*uint16)(unsafe.Pointer(&outView[0])),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.valueRows,
		)

		return nil
	default:
		return tensor.ErrDTypeMismatch
	}
}

func runTypedPageGather(
	storage tensor.Tensor,
	out tensor.Tensor,
	pageTable []int32,
	config pageGatherKernelConfig,
) error {
	switch storage.DType() {
	case dtype.Float32:
		storageView, err := storage.Float32Native()
		if err != nil {
			return err
		}

		outView, err := out.Float32Native()
		if err != nil {
			return err
		}

		pageGatherF32Kernel.kernel(
			&storageView[0], &pageTable[0], &outView[0],
			config.pageCount, config.pageSize, config.inner, config.outRows,
		)

		return nil
	case dtype.Float16:
		storageView, err := storage.Float16Native()
		if err != nil {
			return err
		}

		outView, err := out.Float16Native()
		if err != nil {
			return err
		}

		pageGatherU16Kernel.kernel(
			(*uint16)(unsafe.Pointer(&storageView[0])),
			&pageTable[0],
			(*uint16)(unsafe.Pointer(&outView[0])),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.outRows,
		)

		return nil
	case dtype.BFloat16:
		storageView, err := storage.BFloat16Native()
		if err != nil {
			return err
		}

		outView, err := out.BFloat16Native()
		if err != nil {
			return err
		}

		pageGatherU16Kernel.kernel(
			(*uint16)(unsafe.Pointer(&storageView[0])),
			&pageTable[0],
			(*uint16)(unsafe.Pointer(&outView[0])),
			config.pageCount,
			config.pageSize,
			config.inner,
			config.outRows,
		)

		return nil
	default:
		return tensor.ErrDTypeMismatch
	}
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
