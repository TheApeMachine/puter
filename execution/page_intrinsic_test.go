package execution

import (
	"testing"

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
