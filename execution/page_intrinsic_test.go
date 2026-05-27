package execution

import (
	"context"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunPageWriteIntrinsicPublishesStateStorage(testingObject *testing.T) {
	convey.Convey("Given stacked paged KV storage", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		storage := uploadFloatSliceWithShape(
			testingObject,
			memory,
			make([]float32, 12),
			[]int{2, 3, 2, 1, 1},
		)
		values := uploadFloatSliceWithShape(
			testingObject,
			memory,
			[]float32{7, 8},
			[]int{2, 1, 1},
		)
		pageIDs := uploadInt32Slice(testingObject, memory, []int32{1, 1})
		offsets := uploadInt32Slice(testingObject, memory, []int32{0, 1})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)

		dispatcher.values.set("key_pages", storage)
		dispatcher.values.set("values", values)
		dispatcher.values.set("page_ids", pageIDs)
		dispatcher.values.set("offsets", offsets)

		writeResolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "write",
				Op:     "state.page_write",
				Inputs: []string{"key_pages", "values", "page_ids", "offsets"},
				Attributes: map[string]any{
					"page_size":   2,
					"layer_index": 1,
				},
			},
		}

		convey.Convey("It should publish the full state tensor for downstream consumers", func() {
			err := runPageWriteIntrinsic(writeResolver)

			convey.So(err, convey.ShouldBeNil)

			raw, ok := dispatcher.values.get("write")
			convey.So(ok, convey.ShouldBeTrue)

			written, ok := raw.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(written.Shape().Dims(), convey.ShouldResemble, []int{2, 3, 2, 1, 1})

			storageView, err := storage.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(storageView[8], convey.ShouldEqual, float32(7))
			convey.So(storageView[9], convey.ShouldEqual, float32(8))
		})
	})
}

func TestRunPageWriteIntrinsicDispatchesDeviceStorage(testingObject *testing.T) {
	convey.Convey("Given device-resident paged KV storage", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		storagePointer := unsafe.Pointer(uintptr(0x1000))
		valuesPointer := unsafe.Pointer(uintptr(0x2000))
		pageIDsPointer := unsafe.Pointer(uintptr(0x3000))
		offsetsPointer := unsafe.Pointer(uintptr(0x4000))

		storage := newDispatchTestTensor(testingObject, []int{2, 3, 2, 1, 1}, dtype.Float32, storagePointer)
		values := newDispatchTestTensor(testingObject, []int{2, 1, 1}, dtype.Float32, valuesPointer)
		pageIDs := newDispatchTestTensor(testingObject, []int{2}, dtype.Int32, pageIDsPointer)
		offsets := newDispatchTestTensor(testingObject, []int{2}, dtype.Int32, offsetsPointer)
		deviceBackend := &recordingPageDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)

		dispatcher.values.set("key_pages", storage)
		dispatcher.values.set("values", values)
		dispatcher.values.set("page_ids", pageIDs)
		dispatcher.values.set("offsets", offsets)

		resolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "write",
				Op:     "state.page_write",
				Inputs: []string{"key_pages", "values", "page_ids", "offsets"},
				Attributes: map[string]any{
					"page_size":   2,
					"layer_index": 1,
				},
			},
		}

		convey.Convey("It should call the page write device hook for the selected layer", func() {
			err := runPageWriteIntrinsic(resolver)

			convey.So(err, convey.ShouldBeNil)
			convey.So(deviceBackend.write.storage, convey.ShouldEqual, storagePointer)
			convey.So(deviceBackend.write.values, convey.ShouldEqual, valuesPointer)
			convey.So(deviceBackend.write.pageIDs, convey.ShouldEqual, pageIDsPointer)
			convey.So(deviceBackend.write.offsets, convey.ShouldEqual, offsetsPointer)
			convey.So(deviceBackend.write.output, convey.ShouldEqual, storagePointer)
			convey.So(deviceBackend.write.pageCount, convey.ShouldEqual, 3)
			convey.So(deviceBackend.write.pageSize, convey.ShouldEqual, 2)
			convey.So(deviceBackend.write.inner, convey.ShouldEqual, 1)
			convey.So(deviceBackend.write.valueRows, convey.ShouldEqual, 2)
			convey.So(deviceBackend.write.storageOffset, convey.ShouldEqual, 6)
			convey.So(deviceBackend.write.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

func TestRunPageGatherIntrinsicDispatchesDeviceStorage(testingObject *testing.T) {
	convey.Convey("Given device-resident paged KV storage and live KV length", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		storagePointer := unsafe.Pointer(uintptr(0x5000))
		pageTablePointer := unsafe.Pointer(uintptr(0x6000))

		storage := newDispatchTestTensor(testingObject, []int{2, 3, 2, 1, 1}, dtype.Float32, storagePointer)
		pageTable := newDispatchTestTensor(testingObject, []int{1}, dtype.Int32, pageTablePointer)
		deviceBackend := &recordingPageDevice{}
		dispatcher := newTestDispatcher(deviceBackend, memory)
		dispatcher.launchBindings = ir.SymbolMap{"KV": 1}

		dispatcher.values.set("key_pages", storage)
		dispatcher.values.set("page_table", pageTable)

		gatherShape, err := tensor.NewShape([]int{2, 1, 1})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: gatherShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "gather",
				Op:     "state.page_gather",
				Inputs: []string{"key_pages", "page_table"},
				Attributes: map[string]any{
					"page_size":   2,
					"layer_index": 1,
				},
			},
		}

		convey.Convey("It should cap gathered rows by launch KV and layer offset", func() {
			err := runPageGatherIntrinsic(resolver)

			convey.So(err, convey.ShouldBeNil)
			convey.So(deviceBackend.gather.storage, convey.ShouldEqual, storagePointer)
			convey.So(deviceBackend.gather.pageTable, convey.ShouldEqual, pageTablePointer)
			convey.So(deviceBackend.gather.pageCount, convey.ShouldEqual, 3)
			convey.So(deviceBackend.gather.pageSize, convey.ShouldEqual, 2)
			convey.So(deviceBackend.gather.inner, convey.ShouldEqual, 1)
			convey.So(deviceBackend.gather.outRows, convey.ShouldEqual, 1)
			convey.So(deviceBackend.gather.storageOffset, convey.ShouldEqual, 6)
			convey.So(deviceBackend.gather.format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type dispatchTestTensor struct {
	shape         tensor.Shape
	elementFormat dtype.DType
	pointer       unsafe.Pointer
	byteCount     int
	rawBytes      []byte
}

func newDispatchTestTensor(
	testingObject *testing.T,
	dimensions []int,
	elementFormat dtype.DType,
	pointer unsafe.Pointer,
) *dispatchTestTensor {
	return newDispatchTestTensorWithRaw(testingObject, dimensions, elementFormat, pointer, nil)
}

func newDispatchTestTensorWithRaw(
	testingObject *testing.T,
	dimensions []int,
	elementFormat dtype.DType,
	pointer unsafe.Pointer,
	rawBytes []byte,
) *dispatchTestTensor {
	testingObject.Helper()

	shape, err := tensor.NewShape(dimensions)
	if err != nil {
		testingObject.Fatalf("newDispatchTestTensor: shape: %v", err)
	}

	byteCount, err := shape.Bytes(elementFormat)
	if err != nil {
		testingObject.Fatalf("newDispatchTestTensor: bytes: %v", err)
	}

	return &dispatchTestTensor{
		shape:         shape,
		elementFormat: elementFormat,
		pointer:       pointer,
		byteCount:     byteCount,
		rawBytes:      append([]byte(nil), rawBytes...),
	}
}

func (resident *dispatchTestTensor) Shape() tensor.Shape       { return resident.shape }
func (resident *dispatchTestTensor) DType() dtype.DType        { return resident.elementFormat }
func (resident *dispatchTestTensor) Layout() tensor.Layout     { return tensor.LayoutDense }
func (resident *dispatchTestTensor) Location() tensor.Location { return tensor.Metal }
func (resident *dispatchTestTensor) Len() int                  { return resident.shape.Len() }
func (resident *dispatchTestTensor) Bytes() int                { return resident.byteCount }
func (resident *dispatchTestTensor) Close() error              { return nil }
func (resident *dispatchTestTensor) DispatchPointer() unsafe.Pointer {
	return resident.pointer
}
func (resident *dispatchTestTensor) Slice(start, length int) (tensor.Tensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}
func (resident *dispatchTestTensor) Reshape(dimensions []int) (tensor.Tensor, error) {
	shape, err := tensor.NewShape(dimensions)
	if err != nil {
		return nil, err
	}

	byteCount, err := shape.Bytes(resident.elementFormat)
	if err != nil {
		return nil, err
	}

	if byteCount != resident.byteCount {
		return nil, tensor.ErrShapeMismatch
	}

	return &dispatchTestTensor{
		shape:         shape,
		elementFormat: resident.elementFormat,
		pointer:       resident.pointer,
		byteCount:     byteCount,
		rawBytes:      append([]byte(nil), resident.rawBytes...),
	}, nil
}
func (resident *dispatchTestTensor) Float64Native() ([]float64, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Float32Native() ([]float32, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Float16Native() ([]dtype.F16, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) BFloat16Native() ([]dtype.BF16, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Int64Native() ([]int64, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Int32Native() ([]int32, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Int16Native() ([]int16, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Int8Native() ([]int8, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Uint64Native() ([]uint64, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Uint32Native() ([]uint32, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Uint16Native() ([]uint16, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Uint8Native() ([]uint8, error) {
	return nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) BoolNative() (tensor.BitVector, error) {
	return tensor.BitVector{}, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) Int4Native() (tensor.Int4Vector, error) {
	return tensor.Int4Vector{}, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) RawBytes() (dtype.DType, []byte, error) {
	if resident.rawBytes != nil {
		return resident.elementFormat, append([]byte(nil), resident.rawBytes...), nil
	}

	return dtype.Invalid, nil, tensor.ErrDTypeMismatch
}
func (resident *dispatchTestTensor) State() tensor.State { return tensor.StateReady }
func (resident *dispatchTestTensor) Sync(ctx context.Context) error {
	return ctx.Err()
}
func (resident *dispatchTestTensor) Ready() <-chan struct{} {
	ready := make(chan struct{})
	close(ready)
	return ready
}
func (resident *dispatchTestTensor) RequiresGrad() bool { return false }
func (resident *dispatchTestTensor) SetRequiresGrad(yes bool) error {
	return nil
}
func (resident *dispatchTestTensor) Grad() (tensor.Tensor, error) {
	return nil, tensor.ErrNoAutograd
}
func (resident *dispatchTestTensor) GradFn() tensor.GradFn { return nil }

type pageWriteCall struct {
	storage       unsafe.Pointer
	values        unsafe.Pointer
	pageIDs       unsafe.Pointer
	offsets       unsafe.Pointer
	output        unsafe.Pointer
	pageCount     int
	pageSize      int
	inner         int
	valueRows     int
	storageOffset int
	format        dtype.DType
}

type pageGatherCall struct {
	storage       unsafe.Pointer
	pageTable     unsafe.Pointer
	output        unsafe.Pointer
	pageCount     int
	pageSize      int
	inner         int
	outRows       int
	storageOffset int
	format        dtype.DType
}

type recordingPageDevice struct {
	noopDeviceBackend
	write  pageWriteCall
	gather pageGatherCall
}

func (recorder *recordingPageDevice) PageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageCount, pageSize, inner, valueRows, storageOffset int,
	format dtype.DType,
) {
	recorder.write = pageWriteCall{
		storage:       storage,
		values:        values,
		pageIDs:       pageIDs,
		offsets:       offsets,
		output:        output,
		pageCount:     pageCount,
		pageSize:      pageSize,
		inner:         inner,
		valueRows:     valueRows,
		storageOffset: storageOffset,
		format:        format,
	}
}

func (recorder *recordingPageDevice) PageGather(
	storage, pageTable, output unsafe.Pointer,
	pageCount, pageSize, inner, outRows, storageOffset int,
	format dtype.DType,
) {
	recorder.gather = pageGatherCall{
		storage:       storage,
		pageTable:     pageTable,
		output:        output,
		pageCount:     pageCount,
		pageSize:      pageSize,
		inner:         inner,
		outRows:       outRows,
		storageOffset: storageOffset,
		format:        format,
	}
}

func TestRunPageGatherIntrinsicReadsPageWriteOutput(testingObject *testing.T) {
	convey.Convey("Given page write output feeding page gather", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		storage := uploadFloatSliceWithShape(
			testingObject,
			memory,
			make([]float32, 12),
			[]int{2, 3, 2, 1, 1},
		)
		values := uploadFloatSliceWithShape(
			testingObject,
			memory,
			[]float32{7, 8},
			[]int{2, 1, 1},
		)
		pageIDs := uploadInt32Slice(testingObject, memory, []int32{1, 1})
		offsets := uploadInt32Slice(testingObject, memory, []int32{0, 1})
		pageTable := uploadInt32Slice(testingObject, memory, []int32{1})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.launchBindings = ir.SymbolMap{"KV": 2}

		dispatcher.values.set("key_pages", storage)
		dispatcher.values.set("values", values)
		dispatcher.values.set("page_ids", pageIDs)
		dispatcher.values.set("offsets", offsets)
		dispatcher.values.set("page_table", pageTable)

		writeResolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "write",
				Op:     "state.page_write",
				Inputs: []string{"key_pages", "values", "page_ids", "offsets"},
				Attributes: map[string]any{
					"page_size":   2,
					"layer_index": 1,
				},
			},
		}

		gatherShape, err := tensor.NewShape([]int{2, 1, 1})
		convey.So(err, convey.ShouldBeNil)

		gatherResolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: gatherShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "gather",
				Op:     "state.page_gather",
				Inputs: []string{"write", "page_table"},
				Attributes: map[string]any{
					"page_size":   2,
					"layer_index": 1,
				},
			},
		}

		convey.Convey("It should gather the rows written through the state tensor", func() {
			err := runPageWriteIntrinsic(writeResolver)
			convey.So(err, convey.ShouldBeNil)

			err = runPageGatherIntrinsic(gatherResolver)
			convey.So(err, convey.ShouldBeNil)

			output, err := dispatcher.values.tensor("gather")
			convey.So(err, convey.ShouldBeNil)

			outputView, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(outputView, convey.ShouldResemble, []float32{7, 8})
		})
	})
}
