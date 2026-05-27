package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunBoundNodeUsesShapeReshapeBind(testingObject *testing.T) {
	convey.Convey("Given shape.reshape is declared with a YAML bind", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2, 3,
			4, 5, 6,
		}, []int{2, 3})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.values.set("x", input)

		node := &ast.GraphNode{
			ID:     "reshaped",
			Op:     "shape.reshape",
			Inputs: []string{"x"},
			Attributes: map[string]any{
				"shape": []any{3, 2},
			},
		}

		err := dispatcher.runNode(node)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should store a reshaped tensor view", func() {
			output, err := dispatcher.values.tensor("reshaped")
			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{3, 2})

			values, err := output.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{1, 2, 3, 4, 5, 6})
		})
	})
}

func TestRunReshapeIntrinsicUsesLiveLaunchPrefix(testingObject *testing.T) {
	convey.Convey("Given a max-sized workspace tensor with a live sequence prefix", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
		}, []int{1, 6, 4})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.maxBindings = ir.SymbolMap{"N": 6}
		dispatcher.launchBindings = ir.SymbolMap{"N": 3}
		dispatcher.values.set("hidden", input)

		outputShape, err := tensor.NewShape([]int{1, 3, 2, 2})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: outputShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "heads",
				Op:     "shape.view_as_heads",
				Inputs: []string{"hidden"},
			},
		}

		output, err := runReshapeIntrinsic(resolver)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should reshape only the live prefix", func() {
			tensorOutput, ok := output.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorOutput.Shape().Dims(), convey.ShouldResemble, []int{1, 3, 2, 2})

			values, err := tensorOutput.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})
		})
	})
}

func TestRunReshapeIntrinsicKeepsStaticDimensionsMatchingPlannerMax(testingObject *testing.T) {
	convey.Convey("Given a static hidden dimension equals a planner max binding", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := newDispatchTestTensor(
			testingObject,
			[]int{1, 1024, 4096},
			dtype.Float32,
			unsafe.Pointer(uintptr(0x9000)),
		)
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.maxBindings = ir.SymbolMap{"N": 4096}
		dispatcher.launchBindings = ir.SymbolMap{"N": 1024}
		dispatcher.workspaces = &WorkspaceMap{
			inputTypes: map[string]map[string][]ir.PortType{
				"test": {
					"heads": {
						{
							DType: dtype.Float32,
							ShapeSchema: ir.ShapeSchema{
								Dimensions: []ir.Dimension{
									{Static: 1},
									{Static: 1024},
									{Static: 4096},
								},
							},
						},
					},
				},
			},
		}
		dispatcher.values.set("hidden", input)

		outputShape, err := tensor.NewShape([]int{1, 1024, 16, 256})
		convey.So(err, convey.ShouldBeNil)

		resolver := &bindResolver{
			dispatcher:  dispatcher,
			outputShape: outputShape,
			outputDType: dtype.Float32,
			node: &ast.GraphNode{
				ID:     "heads",
				Op:     "shape.view_as_heads",
				Inputs: []string{"hidden"},
			},
		}

		output, err := runReshapeIntrinsic(resolver)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should not rewrite the hidden dimension to the live sequence length", func() {
			tensorOutput, ok := output.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(tensorOutput.Shape().Dims(), convey.ShouldResemble, []int{1, 1024, 16, 256})
		})
	})
}

func TestLiveShapeIsContiguousPrefix(testingObject *testing.T) {
	convey.Convey("Given planned and live dimensions", testingObject, func() {
		convey.Convey("It should accept a sequence prefix for batch one", func() {
			ok := liveShapeIsContiguousPrefix([]int{1, 6, 4}, []int{1, 3, 4})

			convey.So(ok, convey.ShouldBeTrue)
		})

		convey.Convey("It should reject a middle-dimension prefix across multiple rows", func() {
			ok := liveShapeIsContiguousPrefix([]int{2, 6, 4}, []int{2, 3, 4})

			convey.So(ok, convey.ShouldBeFalse)
		})
	})
}
