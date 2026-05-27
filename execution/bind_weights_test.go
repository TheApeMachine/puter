package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunBoundNodeUsesSlicedTransposedWeightBind(t *testing.T) {
	convey.Convey("Given projection.linear binds to a packed checkpoint slice", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		inputTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 2*3), []int{2, 3})
		weightTensor := uploadFloatSliceWithShape(t, memory, make([]float32, 3*2), []int{3, 2})
		weightStore := &recordingSlicedWeightStore{transposedSlice: weightTensor}
		recorder := &recordingDevice{}
		dispatcher := newTestDispatcher(recorder, memory)
		dispatcher.weights = weightStore
		dispatcher.values.set("x", inputTensor)

		node := &ast.GraphNode{
			ID:     "linear",
			Op:     "projection.linear",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"in_features":  3,
				"out_features": 2,
			},
			Weights: &ast.BoundWeight{
				TensorName: "packed.weight",
				Slice: &ast.WeightSlice{
					Axis:  "output",
					Start: 3,
				},
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The weight store should receive the inferred slice end", func() {
			convey.So(weightStore.axis, convey.ShouldEqual, "output")
			convey.So(weightStore.start, convey.ShouldEqual, int64(3))
			convey.So(weightStore.end, convey.ShouldEqual, int64(5))
		})

		convey.Convey("The router should use the sliced weight shape", func() {
			convey.So(recorder.matmulCalls, convey.ShouldHaveLength, 1)
			convey.So(recorder.matmulCalls[0].rows, convey.ShouldEqual, 2)
			convey.So(recorder.matmulCalls[0].inner, convey.ShouldEqual, 3)
			convey.So(recorder.matmulCalls[0].cols, convey.ShouldEqual, 2)
			convey.So(recorder.matmulCalls[0].format, convey.ShouldEqual, dtype.Float32)
		})
	})
}

type recordingSlicedWeightStore struct {
	transposedSlice tensor.Tensor
	name            string
	axis            string
	start           int64
	end             int64
}

func (store *recordingSlicedWeightStore) Lookup(name string) (tensor.Tensor, error) {
	_ = name

	return nil, ErrWeightNotFound
}

func (store *recordingSlicedWeightStore) LookupTransposedSlice(
	name, axis string,
	start, end int64,
) (tensor.Tensor, error) {
	store.name = name
	store.axis = axis
	store.start = start
	store.end = end

	return store.transposedSlice, nil
}
