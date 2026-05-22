package shape

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestRunPageWriteGather(testingObject *testing.T) {
	for _, length := range parity.Lengths {
		length := length

		testingObject.Run(fmt.Sprintf("N=%d", length), func(testingObject *testing.T) {
			convey.Convey("Given paged storage and logical row values", testingObject, func() {
				runPageWriteGatherParityCase(testingObject, length)
			})
		})
	}
}

func runPageWriteGatherParityCase(testingObject *testing.T, length int) {
	const pageSize = 4
	const inner = 8

	pageCount := (length + pageSize - 1) / pageSize

	storageShape, err := tensor.NewShape([]int{pageCount, pageSize, inner})
	convey.So(err, convey.ShouldBeNil)

	valueShape, err := tensor.NewShape([]int{length, inner})
	convey.So(err, convey.ShouldBeNil)

	pageTableShape, err := tensor.NewShape([]int{pageCount})
	convey.So(err, convey.ShouldBeNil)

	storage, err := tensor.NewZeroed(storageShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	values, err := tensor.NewZeroed(valueShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	pageIDs, err := tensor.NewZeroed(valueShapeWithoutInner(length), dtype.Int32)
	convey.So(err, convey.ShouldBeNil)

	offsets, err := tensor.NewZeroed(valueShapeWithoutInner(length), dtype.Int32)
	convey.So(err, convey.ShouldBeNil)

	pageSizeTensor, err := newInt32ScalarTensor(pageSize)
	convey.So(err, convey.ShouldBeNil)

	written, err := tensor.NewZeroed(storageShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	populatePageWriteInputs(values, pageIDs, offsets, pageSize)

	err = RunPageWrite(storage, values, pageIDs, offsets, pageSizeTensor, written)
	convey.So(err, convey.ShouldBeNil)

	pageTable, err := tensor.NewZeroed(pageTableShape, dtype.Int32)
	convey.So(err, convey.ShouldBeNil)
	populatePageTable(pageTable)

	gathered, err := tensor.NewZeroed(valueShape, dtype.Float32)
	convey.So(err, convey.ShouldBeNil)

	err = RunPageGather(written, pageTable, pageSizeTensor, gathered)
	convey.So(err, convey.ShouldBeNil)

	got, err := gathered.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	want, err := values.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	parity.AssertFloat32SlicesWithinULP(testingObject, got, want, 0)
}

func valueShapeWithoutInner(length int) tensor.Shape {
	shape, err := tensor.NewShape([]int{length})

	if err != nil {
		panic(err)
	}

	return shape
}

func populatePageWriteInputs(values tensor.Tensor, pageIDs tensor.Tensor, offsets tensor.Tensor, pageSize int) {
	valueView, err := values.Float32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range valueView {
		valueView[index] = float32(index)*0.01 - 2
	}

	pageIDView, err := pageIDs.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	offsetView, err := offsets.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range pageIDView {
		pageIDView[index] = int32(index / pageSize)
		offsetView[index] = int32(index % pageSize)
	}
}

func populatePageTable(pageTable tensor.Tensor) {
	pageTableView, err := pageTable.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range pageTableView {
		pageTableView[index] = int32(index)
	}
}
