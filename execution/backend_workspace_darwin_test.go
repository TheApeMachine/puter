//go:build darwin && cgo

package execution

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/qpool"
)

func TestAttachWorkspaceUsesMetalResidentWorkspace(testingObject *testing.T) {
	convey.Convey("Given an execution backend pinned to Metal", testingObject, func() {
		workerPool := qpool.NewQ[any](context.Background(), 1, 1, nil)
		defer workerPool.Close()

		devicePool, err := pool.New(context.Background(), workerPool)
		convey.So(err, convey.ShouldBeNil)
		defer devicePool.Close()

		err = devicePool.PinTo(tensor.Metal)
		convey.So(err, convey.ShouldBeNil)

		backend := New(devicePool)
		defer backend.Close()

		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{{ID: "first", Op: "test.first"}},
		}
		topology := &ir.Topology{
			Nodes: []*ir.Node{{
				Name:    "first",
				Outputs: []*ir.Port{workspaceResidentPort(1, 8)},
			}},
			Workspace: ir.WorkspaceLayout{
				Size:  64,
				Align: 64,
				Allocations: []ir.Interval{
					{PortID: 1, Start: 0, End: 0, Offset: 0, Size: 64},
				},
			},
		}

		err = backend.AttachWorkspace("model", graph, topology)

		convey.Convey("It should attach Metal tensors from the planner", func() {
			convey.So(err, convey.ShouldBeNil)

			output, ok := backend.Workspaces().OutputFor("model", "first")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(output.Location(), convey.ShouldEqual, tensor.Metal)
			convey.So(output.Shape().Dims(), convey.ShouldResemble, []int{8})
		})
	})
}
