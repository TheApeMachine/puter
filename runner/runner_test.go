package runner

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/qpool"
)

func TestRunnerCallGraphMatMul(testingObject *testing.T) {
	convey.Convey("Given a matmul compute graph on the host device", testingObject, func() {
		workerPool := qpool.NewQ(context.Background(), 1, 2, qpool.NewConfig())
		defer workerPool.Close()

		devicePool, err := pool.New(context.Background(), workerPool)
		convey.So(err, convey.ShouldBeNil)
		defer devicePool.Close()

		graphRunner := New(devicePool)

		leftShape, err := tensor.NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		rightShape, err := tensor.NewShape([]int{3, 4})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{2, 4})
		convey.So(err, convey.ShouldBeNil)

		left := manifestComputeNode("left", "input", ir.OpInput, leftShape)
		right := manifestComputeNode("right", "input", ir.OpInput, rightShape)
		matmulNode := manifestComputeNode("matmul", "math.matmul", ir.OpMatmul, outputShape)
		setFloat32ValueType(left)
		setFloat32ValueType(right)
		setFloat32ValueType(matmulNode)
		matmulNode.AddInput(left)
		matmulNode.AddInput(right)

		computeGraph := ir.NewGraph()
		computeGraph.AddNode(left)
		computeGraph.AddNode(right)
		computeGraph.AddNode(matmulNode)

		leftValues := []float32{1, 2, 3, 4, 5, 6}
		rightValues := make([]float32, 12)

		for index := range rightValues {
			rightValues[index] = float32(index + 1)
		}

		result, err := graphRunner.CallGraph(context.Background(), runtime.GraphCallRequest{
			GraphName: "demo",
			Graph: &ast.Graph{
				Inputs:  []string{"left", "right"},
				Outputs: map[string]string{"out": "matmul"},
			},
			Compute: computeGraph,
			Inputs: map[string]any{
				"left":  leftValues,
				"right": rightValues,
			},
		})

		convey.So(err, convey.ShouldBeNil)

		output, ok := result.Outputs["out"].([]float32)
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(len(output), convey.ShouldEqual, 8)
		convey.So(output[0], convey.ShouldAlmostEqual, expectedMatMul(leftValues, rightValues, 0, 0), 1e-4)
	})
}

func manifestComputeNode(
	nodeID string,
	manifestOp string,
	opType ir.OpType,
	shape tensor.Shape,
) *ir.Node {
	node := ir.NewNode(nodeID, opType, shape)
	node.SetOperationID(ir.OpID(manifestOp))

	return node
}

func setFloat32ValueType(node *ir.Node) {
	valueType := node.ValueType()
	valueType.DType = dtype.Float32
	valueType.Precision = dtype.Float32
	node.SetValueType(valueType)
}

func expectedMatMul(
	left []float32,
	right []float32,
	row int,
	col int,
) float32 {
	inner := 3
	cols := 4
	total := float32(0)

	for index := range inner {
		leftValue := left[row*inner+index]
		rightValue := right[index*cols+col]
		total += leftValue * rightValue
	}

	return total
}

func TestRunnerCallGraphRequiresComputeGraph(testingObject *testing.T) {
	convey.Convey("Given a graph call without compute IR", testingObject, func() {
		devicePool, err := pool.New(context.Background(), nil)
		convey.So(err, convey.ShouldBeNil)
		defer devicePool.Close()

		_, err = New(devicePool).CallGraph(context.Background(), runtime.GraphCallRequest{
			GraphName: "demo",
			Graph:     &ast.Graph{},
		})

		convey.So(err, convey.ShouldNotBeNil)
	})
}

func TestRunnerWeightCache(testingObject *testing.T) {
	convey.Convey("Given a runner and a compute backend", testingObject, func() {
		devicePool, err := pool.New(context.Background(), nil)
		convey.So(err, convey.ShouldBeNil)
		defer devicePool.Close()

		memory, _, err := devicePool.ComputeMemory()
		convey.So(err, convey.ShouldBeNil)

		graphRunner := New(devicePool)
		defer graphRunner.Close()

		convey.Convey("It should reuse the same resident weight cache across calls", func() {
			first := graphRunner.weightCache(memory)
			second := graphRunner.weightCache(memory)

			convey.So(first, convey.ShouldEqual, second)
		})
	})
}

func BenchmarkRunnerCallGraphMatMul(benchmark *testing.B) {
	workerPool := qpool.NewQ(context.Background(), 1, 2, qpool.NewConfig())
	defer workerPool.Close()

	devicePool, err := pool.New(context.Background(), workerPool)

	if err != nil {
		benchmark.Fatal(err)
	}

	defer devicePool.Close()

	graphRunner := New(devicePool)
	leftShape, _ := tensor.NewShape([]int{64, 64})
	rightShape, _ := tensor.NewShape([]int{64, 64})
	outputShape, _ := tensor.NewShape([]int{64, 64})

	left := manifestComputeNode("left", "input", ir.OpInput, leftShape)
	right := manifestComputeNode("right", "input", ir.OpInput, rightShape)
	matmulNode := manifestComputeNode("matmul", "math.matmul", ir.OpMatmul, outputShape)
	matmulNode.AddInput(left)
	matmulNode.AddInput(right)

	computeGraph := ir.NewGraph()
	computeGraph.AddNode(left)
	computeGraph.AddNode(right)
	computeGraph.AddNode(matmulNode)

	leftValues := make([]float32, 64*64)
	rightValues := make([]float32, 64*64)

	for index := range leftValues {
		leftValues[index] = float32(index) * 0.001
		rightValues[index] = float32(index) * 0.002
	}

	request := runtime.GraphCallRequest{
		GraphName: "demo",
		Graph: &ast.Graph{
			Inputs:  []string{"left", "right"},
			Outputs: map[string]string{"out": "matmul"},
		},
		Compute: computeGraph,
		Inputs: map[string]any{
			"left":  leftValues,
			"right": rightValues,
		},
	}

	for benchmark.Loop() {
		result, callErr := graphRunner.CallGraph(context.Background(), request)

		if callErr != nil {
			benchmark.Fatal(callErr)
		}

		_ = result
	}
}
