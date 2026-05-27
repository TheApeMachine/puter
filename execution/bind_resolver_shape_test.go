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
