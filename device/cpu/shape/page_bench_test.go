package shape

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkRunPageWriteGather(benchmark *testing.B) {
	const length = 8192
	const pageSize = 16
	const inner = 64

	pageCount := (length + pageSize - 1) / pageSize
	storageShape, _ := tensor.NewShape([]int{pageCount, pageSize, inner})
	valueShape, _ := tensor.NewShape([]int{length, inner})
	indexShape, _ := tensor.NewShape([]int{length})
	tableShape, _ := tensor.NewShape([]int{pageCount})

	storage, _ := tensor.NewZeroed(storageShape, dtype.Float32)
	values, _ := tensor.NewZeroed(valueShape, dtype.Float32)
	pageIDs, _ := tensor.NewZeroed(indexShape, dtype.Int32)
	offsets, _ := tensor.NewZeroed(indexShape, dtype.Int32)
	pageSizeTensor, _ := newInt32ScalarTensor(pageSize)
	written, _ := tensor.NewZeroed(storageShape, dtype.Float32)
	pageTable, _ := tensor.NewZeroed(tableShape, dtype.Int32)
	gathered, _ := tensor.NewZeroed(valueShape, dtype.Float32)

	populatePageWriteInputsForBenchmark(benchmark, values, pageIDs, offsets, pageSize)
	populatePageTableForBenchmark(benchmark, pageTable)

	benchmark.ReportAllocs()
	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = RunPageWrite(storage, values, pageIDs, offsets, pageSizeTensor, written)
		_ = RunPageGather(written, pageTable, pageSizeTensor, gathered)
	}
}

func populatePageWriteInputsForBenchmark(
	benchmark *testing.B,
	values tensor.Tensor,
	pageIDs tensor.Tensor,
	offsets tensor.Tensor,
	pageSize int,
) {
	benchmark.Helper()

	valueView, err := values.Float32Native()
	if err != nil {
		benchmark.Fatal(err)
	}

	for index := range valueView {
		valueView[index] = float32(index)*0.01 - 2
	}

	pageIDView, err := pageIDs.Int32Native()
	if err != nil {
		benchmark.Fatal(err)
	}

	offsetView, err := offsets.Int32Native()
	if err != nil {
		benchmark.Fatal(err)
	}

	for index := range pageIDView {
		pageIDView[index] = int32(index / pageSize)
		offsetView[index] = int32(index % pageSize)
	}
}

func populatePageTableForBenchmark(benchmark *testing.B, pageTable tensor.Tensor) {
	benchmark.Helper()

	pageTableView, err := pageTable.Int32Native()
	if err != nil {
		benchmark.Fatal(err)
	}

	for index := range pageTableView {
		pageTableView[index] = int32(index)
	}
}
