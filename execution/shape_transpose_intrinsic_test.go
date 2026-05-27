package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunBoundNodeUsesShapeTransposeBind(testingObject *testing.T) {
	convey.Convey("Given shape.transpose is declared with a YAML bind", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2, 3,
			4, 5, 6,
		}, []int{2, 3})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "transposed",
			Op:     "shape.transpose",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"dim0": 0,
				"dim1": 1,
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should swap the configured dimensions", func() {
			output, err := dispatcher.values.tensor("transposed")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{3, 2})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{1, 4, 2, 5, 3, 6})
		})
	})
}
