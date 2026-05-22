package shape

import (
	"fmt"
	"runtime"
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

func TestRunPageWriteGatherReducedPrecision(testingObject *testing.T) {
	for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		for _, length := range parity.Lengths {
			length := length

			testingObject.Run(fmt.Sprintf("%s/N=%d", storageDType, length), func(testingObject *testing.T) {
				convey.Convey("Given reduced-precision paged storage and logical row values", testingObject, func() {
					runPageWriteGatherReducedPrecisionCase(length, storageDType)
				})
			})
		}
	}
}

func TestPageKernelISASelection(testingObject *testing.T) {
	convey.Convey("Given page kernels selected for this CPU", testingObject, func() {
		if runtime.GOARCH == "arm64" {
			convey.So(pageWriteF32Kernel.name, convey.ShouldEqual, "neon")
			convey.So(pageGatherF32Kernel.name, convey.ShouldEqual, "neon")
			convey.So(pageWriteU16Kernel.name, convey.ShouldEqual, "neon")
			convey.So(pageGatherU16Kernel.name, convey.ShouldEqual, "neon")
		}

		if runtime.GOARCH == "amd64" {
			convey.So(pageWriteF32Kernel.name, convey.ShouldNotEqual, "generic")
			convey.So(pageGatherF32Kernel.name, convey.ShouldNotEqual, "generic")
			convey.So(pageWriteU16Kernel.name, convey.ShouldNotEqual, "generic")
			convey.So(pageGatherU16Kernel.name, convey.ShouldNotEqual, "generic")
		}
	})
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

func runPageWriteGatherReducedPrecisionCase(length int, storageDType dtype.DType) {
	const pageSize = 4
	const inner = 8

	pageCount := (length + pageSize - 1) / pageSize

	storageShape, err := tensor.NewShape([]int{pageCount, pageSize, inner})
	convey.So(err, convey.ShouldBeNil)

	valueShape, err := tensor.NewShape([]int{length, inner})
	convey.So(err, convey.ShouldBeNil)

	pageTableShape, err := tensor.NewShape([]int{pageCount})
	convey.So(err, convey.ShouldBeNil)

	storage, err := tensor.NewZeroed(storageShape, storageDType)
	convey.So(err, convey.ShouldBeNil)

	values, err := tensor.NewZeroed(valueShape, storageDType)
	convey.So(err, convey.ShouldBeNil)

	pageIDs, err := tensor.NewZeroed(valueShapeWithoutInner(length), dtype.Int32)
	convey.So(err, convey.ShouldBeNil)

	offsets, err := tensor.NewZeroed(valueShapeWithoutInner(length), dtype.Int32)
	convey.So(err, convey.ShouldBeNil)

	pageSizeTensor, err := newInt32ScalarTensor(pageSize)
	convey.So(err, convey.ShouldBeNil)

	written, err := tensor.NewZeroed(storageShape, storageDType)
	convey.So(err, convey.ShouldBeNil)

	populateReducedPrecisionValues(values, storageDType)
	populatePageIndices(pageIDs, offsets, pageSize)

	err = RunPageWrite(storage, values, pageIDs, offsets, pageSizeTensor, written)
	convey.So(err, convey.ShouldBeNil)

	pageTable, err := tensor.NewZeroed(pageTableShape, dtype.Int32)
	convey.So(err, convey.ShouldBeNil)
	populatePageTable(pageTable)

	gathered, err := tensor.NewZeroed(valueShape, storageDType)
	convey.So(err, convey.ShouldBeNil)

	err = RunPageGather(written, pageTable, pageSizeTensor, gathered)
	convey.So(err, convey.ShouldBeNil)

	assertReducedPrecisionEqual(gathered, values, storageDType)
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

	populatePageIndices(pageIDs, offsets, pageSize)
}

func populatePageIndices(pageIDs tensor.Tensor, offsets tensor.Tensor, pageSize int) {
	pageIDView, err := pageIDs.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	offsetView, err := offsets.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range pageIDView {
		pageIDView[index] = int32(index / pageSize)
		offsetView[index] = int32(index % pageSize)
	}
}

func populateReducedPrecisionValues(values tensor.Tensor, storageDType dtype.DType) {
	switch storageDType {
	case dtype.Float16:
		view, err := values.Float16Native()
		convey.So(err, convey.ShouldBeNil)

		for index := range view {
			view[index] = dtype.F16(0x3c00 + uint16(index%1024))
		}
	case dtype.BFloat16:
		view, err := values.BFloat16Native()
		convey.So(err, convey.ShouldBeNil)

		for index := range view {
			view[index] = dtype.BF16(0x3f80 + uint16(index%1024))
		}
	}
}

func assertReducedPrecisionEqual(got tensor.Tensor, want tensor.Tensor, storageDType dtype.DType) {
	switch storageDType {
	case dtype.Float16:
		gotView, err := got.Float16Native()
		convey.So(err, convey.ShouldBeNil)

		wantView, err := want.Float16Native()
		convey.So(err, convey.ShouldBeNil)
		convey.So(gotView, convey.ShouldResemble, wantView)
	case dtype.BFloat16:
		gotView, err := got.BFloat16Native()
		convey.So(err, convey.ShouldBeNil)

		wantView, err := want.BFloat16Native()
		convey.So(err, convey.ShouldBeNil)
		convey.So(gotView, convey.ShouldResemble, wantView)
	}
}

func populatePageTable(pageTable tensor.Tensor) {
	pageTableView, err := pageTable.Int32Native()
	convey.So(err, convey.ShouldBeNil)

	for index := range pageTableView {
		pageTableView[index] = int32(index)
	}
}
