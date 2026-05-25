package execution

import (
	"context"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
)

/*
TestCallGraphRequiresGraphAndPlan covers the early-return guard that
prevents the dispatcher from running against a malformed request. It
exercises only the public CallGraph contract — no device backend
needed.
*/
func TestCallGraphRequiresGraphAndPlan(t *testing.T) {
	convey.Convey("Given a Backend with no devicePool", t, func() {
		backend := &Backend{}

		_, err := backend.CallGraph(context.Background(), runtime.GraphCallRequest{})

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "device pool is required")
	})
}

/*
TestDispatcherRunsFusedNode runs the dispatcher against a hand-rolled
graph containing one FuseOp node with a codegen-attached CPU kernel.
The kernel implements ReLU(Add(x, y)); the test asserts the output
tensor materializes the expected element-wise result.
*/
func TestDispatcherRunsFusedNode(t *testing.T) {
	convey.Convey("Given a graph with one FuseOp node compiled for CPU", t, func() {
		fusion := &optimizer.FusionAST{
			InputPorts: []string{"x", "y"},
			OutputPort: "result",
			Root: &optimizer.ASTNode{
				Type: optimizer.NodeReLU,
				Children: []*optimizer.ASTNode{
					{
						Type: optimizer.NodeAdd,
						Children: []*optimizer.ASTNode{
							{Type: optimizer.NodeInput, InputIndex: 0},
							{Type: optimizer.NodeInput, InputIndex: 1},
						},
					},
				},
			},
		}

		kernel, err := codegen.EmitCPU(fusion)
		convey.So(err, convey.ShouldBeNil)

		graph := &ast.Graph{
			Inputs:  []string{"x", "y"},
			Outputs: map[string]string{"out": "result"},
			Nodes: []*ast.GraphNode{
				{
					ID:     "result",
					Op:     optimizer.FuseOp,
					Inputs: []string{"x", "y"},
					Attributes: map[string]any{
						optimizer.FuseAttributeAST: fusion,
						codegen.KernelAttribute:    codegen.NewKernelSet([]codegen.Kernel{kernel}),
					},
				},
			},
		}

		memory := tensor.NewHostBackend()

		xTensor := uploadFloat32(t, memory, []float32{1, -3, 5, -2})
		yTensor := uploadFloat32(t, memory, []float32{2, 1, -4, 1})

		plan := &runtime.ExecutionPlan{
			GraphName: "test",
			Layers:    [][]string{{"result"}},
		}

		dispatcher := newDispatcher(
			graph, plan,
			noopDeviceBackend{},
			memory,
			nilWeightStore{},
		)

		dispatcher.values.set("x", xTensor)
		dispatcher.values.set("y", yTensor)

		err = dispatcher.run()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("Then the result tensor contains ReLU(x + y) elementwise", func() {
			result, err := dispatcher.values.tensor("result")
			convey.So(err, convey.ShouldBeNil)

			values, err := result.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{3, 0, 1, 0})
		})
	})
}

/*
TestDispatcherFailsOnUnknownOp verifies the "unsupported op" diagnostic
fires when a graph contains an op string with no handler.
*/
func TestDispatcherFailsOnUnknownOp(t *testing.T) {
	convey.Convey("Given a graph with an unsupported op", t, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{ID: "node", Op: "definitely.not.a.real.op"},
			},
		}

		plan := &runtime.ExecutionPlan{
			GraphName: "test",
			Layers:    [][]string{{"node"}},
		}

		dispatcher := newDispatcher(
			graph, plan,
			noopDeviceBackend{},
			tensor.NewHostBackend(),
			nilWeightStore{},
		)

		err := dispatcher.run()
		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "unsupported op")
	})
}

/*
TestDispatcherFailsOnMissingWeight verifies the weight store's
ErrWeightNotFound is surfaced as a clear, debuggable error.
*/
func TestDispatcherFailsOnMissingWeight(t *testing.T) {
	convey.Convey("Given a graph with a weighted op and no weight store entry", t, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID:      "embed",
					Op:      "embedding.token",
					Inputs:  []string{"tokens"},
					Weights: &ast.BoundWeight{TensorName: "model.embed_tokens.weight"},
				},
			},
		}

		plan := &runtime.ExecutionPlan{
			GraphName: "test",
			Layers:    [][]string{{"embed"}},
		}

		dispatcher := newDispatcher(
			graph, plan,
			noopDeviceBackend{},
			tensor.NewHostBackend(),
			nilWeightStore{},
		)

		dispatcher.values.set("tokens", []int{1, 2, 3})

		err := dispatcher.run()
		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "weight not found")
	})
}

/*
uploadFloat32 is a small test helper that materializes a float32 slice as a
host tensor via the standard Upload path.
*/
func uploadFloat32(t *testing.T, backend tensor.Backend, values []float32) tensor.Tensor {
	t.Helper()

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		t.Fatalf("uploadFloat32 shape: %v", err)
	}

	bytes := make([]byte, len(values)*4)

	for index, value := range values {
		*(*float32)(unsafe.Pointer(&bytes[index*4])) = value
	}

	uploaded, err := backend.Upload(shape, dtype.Float32, bytes)

	if err != nil {
		t.Fatalf("uploadFloat32 upload: %v", err)
	}

	return uploaded
}
