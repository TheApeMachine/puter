package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
TestResolveShapeSubstitutesBindings is the unit-level regression test
for the caramba chat failure:

	attach workspace for graph "model": execution: port 1 shape:
	dim[0] symbol "N" unresolved at workspace materialization

The planner sized port id=1's tensor using bindings {"N": 4096} (the
KV-cache page count from chat.yml), so byte sizes were correct, but
Port.Type.ShapeSchema still carried [N] as a symbolic Dimension. When
the workspace materializer turned that schema into a tensor.Shape it
had no way to substitute the symbol and bailed.

The fix stores the planner's bindings on Topology.Workspace and threads
them into resolveShape. This test asserts that path: a symbolic
dimension with a binding present resolves to the bound value, and one
without a binding still surfaces the original error so callers don't
silently fall back to a zero shape.
*/
func TestResolveShapeSubstitutesBindings(t *testing.T) {
	convey.Convey("Given a symbolic ShapeSchema [N, 4096]", t, func() {
		schema := ir.ShapeSchema{
			Dimensions: []ir.Dimension{
				{Symbol: "N"},
				{Static: 4096},
			},
		}

		convey.Convey("With a binding for N", func() {
			shape, err := resolveShape(schema, dtype.Float32, ir.SymbolMap{"N": 4096})
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{4096, 4096})
		})

		convey.Convey("Without a binding for N", func() {
			_, err := resolveShape(schema, dtype.Float32, ir.SymbolMap{})
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "unresolved at workspace materialization")
		})

		convey.Convey("With nil bindings", func() {
			_, err := resolveShape(schema, dtype.Float32, nil)
			convey.So(err, convey.ShouldNotBeNil)
		})
	})
}

/*
TestWorkspaceMaterializesSymbolicPortViaPlannerBindings is the end-to-
end version of the same fix. It builds a one-node topology whose only
output port has a symbolic shape, runs PlanWorkspace with a binding,
then constructs the Workspace and asserts that the materialized tensor
carries the resolved concrete shape rather than blowing up.

This is the path caramba chat exercises: planner sizes everything, then
the orchestrator hands the planned topology to AttachWorkspace which
calls NewWorkspace which calls preBuildTensors which calls resolveShape.
*/
func TestWorkspaceMaterializesSymbolicPortViaPlannerBindings(t *testing.T) {
	convey.Convey("Given a planner topology with one [N, 4] port and a binding N=8", t, func() {
		port := &ir.Port{
			Type: ir.PortType{
				DType: dtype.Float32,
				ShapeSchema: ir.ShapeSchema{
					Dimensions: []ir.Dimension{
						{Symbol: "N"},
						{Static: 4},
					},
				},
				Layout: ir.LayoutContiguous,
			},
		}

		node := &ir.Node{
			Name:    "producer",
			Outputs: []*ir.Port{port},
		}

		topology := &ir.Topology{
			Nodes: []*ir.Node{node},
		}

		err := ir.PlanWorkspace(topology, ir.PlanWorkspaceOptions{
			Bindings: ir.SymbolMap{"N": 8},
			Align:    64,
		})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("The planner persists the binding on WorkspaceLayout", func() {
			convey.So(topology.Workspace.Bindings["N"], convey.ShouldEqual, int64(8))
		})

		convey.Convey("NewWorkspace materializes the port with concrete dims", func() {
			workspace, err := NewWorkspace(topology)
			convey.So(err, convey.ShouldBeNil)

			defer workspace.Close()

			resolved, err := workspace.TensorByPortID(port.ID)
			convey.So(err, convey.ShouldBeNil)
			convey.So(resolved.Shape().Dims(), convey.ShouldResemble, []int{8, 4})
		})
	})
}

/*
TestAttachWorkspaceCarriesBindingsToMaterializer wires the public
WorkspaceMap.Attach path so we cover the route caramba actually takes:
ast.Graph + planned *ir.Topology → Attach → NewWorkspace →
preBuildTensors. If the bindings stop flowing anywhere along this path
this test fails with the original "unresolved at workspace
materialization" message.
*/
func TestAttachWorkspaceCarriesBindingsToMaterializer(t *testing.T) {
	convey.Convey("Given an ast.Graph + planned topology with one symbolic port", t, func() {
		port := &ir.Port{
			Type: ir.PortType{
				DType: dtype.Float32,
				ShapeSchema: ir.ShapeSchema{
					Dimensions: []ir.Dimension{
						{Symbol: "N"},
						{Static: 4},
					},
				},
				Layout: ir.LayoutContiguous,
			},
		}

		topology := &ir.Topology{
			Nodes: []*ir.Node{{
				Name:    "n0",
				Outputs: []*ir.Port{port},
			}},
		}

		err := ir.PlanWorkspace(topology, ir.PlanWorkspaceOptions{
			Bindings: ir.SymbolMap{"N": 8},
			Align:    64,
		})
		convey.So(err, convey.ShouldBeNil)

		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{{ID: "n0", Op: "test.placeholder"}},
		}

		workspaceMap := NewWorkspaceMap()
		defer workspaceMap.Close()

		err = workspaceMap.Attach("model", graph, topology)
		convey.So(err, convey.ShouldBeNil)
	})
}
