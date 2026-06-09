package execution

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func TestResolveInputDimensionsPrefersProducedRuntimeTensor(testingObject *testing.T) {
	convey.Convey("Given a produced reshape tensor and an aliased planner input port", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := newDispatchTestTensor(
			testingObject,
			[]int{1, 1024, 4096},
			dtype.Float32,
			unsafe.Pointer(uintptr(0x9100)),
		)
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.workspaces = &WorkspaceMap{
			inputTypes: map[string]map[string][]ir.PortType{
				"test": {
					"linear": {
						{
							DType: dtype.Float32,
							ShapeSchema: ir.ShapeSchema{
								Dimensions: []ir.Dimension{
									{Static: 1},
									{Static: 1024},
									{Static: 32},
									{Static: 128},
								},
							},
						},
					},
				},
			},
		}
		dispatcher.values.set("merge", input)

		resolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "linear",
				Op:     "projection.linear",
				Inputs: []string{"merge"},
			},
		}

		dimensions, err := resolver.resolveInputDimensions("0", input)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should use the runtime tensor shape", func() {
			convey.So(dimensions, convey.ShouldResemble, []int{1, 1024, 4096})
		})
	})
}

func TestResolveOutputShapeKeepsStaticOutFeatures(testingObject *testing.T) {
	convey.Convey("Given linear out_features equals a planner max binding", testingObject, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		input := uploadFloatSliceWithShape(testingObject, memory, make([]float32, 1*1024*2560), []int{1, 1024, 2560})
		dispatcher := newTestDispatcher(noopDeviceBackend{}, memory)
		dispatcher.graphName = "test"
		dispatcher.maxBindings = ir.SymbolMap{"N": 1024, "T": 4096}
		dispatcher.launchBindings = ir.SymbolMap{"N": 1024, "T": 1024}
		dispatcher.workspaces = &WorkspaceMap{
			outputTypes: map[string]map[string]ir.PortType{
				"test": {
					"q_proj": {
						DType: dtype.BFloat16,
						ShapeSchema: ir.ShapeSchema{
							Dimensions: []ir.Dimension{
								{Static: 1},
								{Symbol: "N"},
								{Static: 4096},
							},
						},
					},
				},
			},
		}
		dispatcher.values.set("hidden", input)

		resolver := &bindResolver{
			dispatcher: dispatcher,
			node: &ast.GraphNode{
				ID:     "q_proj",
				Op:     "projection.linear",
				Inputs: []string{"hidden"},
				Attributes: map[string]any{
					"in_features":  int64(2560),
					"out_features": int64(4096),
				},
			},
			bind: OperationBind{
				Output: asset.BindOutput{
					Shape: []asset.BindArg{
						{Ref: "input.x.shape", DropTail: 1},
						{Ref: "config.out_features.int"},
					},
					DType: "input.x.dtype",
				},
			},
		}

		shape, err := resolver.resolveOutputShape()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should keep out_features at 4096", func() {
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 1024, 4096})
		})
	})
}
