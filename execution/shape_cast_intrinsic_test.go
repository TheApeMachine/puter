package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunCastIntrinsicConvertsFloat32ToInt32(testingObject *testing.T) {
	convey.Convey("Given a float32 tensor cast to int32", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1.5, 2.0, -3.7,
		}, []int{3})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		outputShape, err := tensor.NewShape([]int{3})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: outputShape,
			outputDType: dtype.Int32,
			node: &ast.GraphNode{
				ID:     "casted",
				Op:     "shape.cast",
				Inputs: []string{"x"},
				Attributes: map[string]any{
					"dtype":      "int32",
					"from_dtype": "float32",
				},
			},
		}

		output, err := runCastIntrinsic(resolver)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should preserve shape and narrow dtypes", func() {
			tensorOutput, ok := output.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorOutput.DType(), convey.ShouldEqual, dtype.Int32)
			convey.So(tensorOutput.Shape().Dims(), convey.ShouldResemble, []int{3})

			_, raw, err := tensorOutput.RawBytes()
			convey.So(err, convey.ShouldBeNil)

			int32Values, err := tensorOutput.Int32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(int32Values, convey.ShouldResemble, []int32{1, 2, -3})
			convey.So(len(raw), convey.ShouldEqual, 12)
		})
	})
}

func TestRunCastIntrinsicIdentityWhenDTypeMatches(testingObject *testing.T) {
	convey.Convey("Given matching source and target dtypes", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{1, 2}, []int{2})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		outputShape, err := tensor.NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: outputShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "casted",
				Op:     "shape.cast",
				Inputs: []string{"x"},
			},
		}

		output, err := runCastIntrinsic(resolver)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should return the live input without copying", func() {
			tensorOutput, ok := output.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorOutput, convey.ShouldEqual, input)
		})
	})
}
