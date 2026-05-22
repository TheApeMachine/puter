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

func TestConcatOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a concat node with two same-rank float inputs", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 8, 9728})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 8, 19456})
		convey.So(err, convey.ShouldBeNil)

		leftNode := manifestComputeNode("gate", "input", ir.OpInput, inputShape)
		rightNode := manifestComputeNode("up", "input", ir.OpInput, inputShape)
		concatNode := manifestComputeNode("gate_up_0", "shape.concat", ir.OpFused, outputShape)
		concatNode.SetAttribute("dim", ir.IntAttribute(2))
		concatNode.AddInput(leftNode)
		concatNode.AddInput(rightNode)
		setFloat32ValueType(leftNode)
		setFloat32ValueType(rightNode)
		setFloat32ValueType(concatNode)

		left, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		right, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("gate", left)
		tensorWorkspace.Store("up", right)

		convey.Convey("It should allocate the concatenated shape", func() {
			shape, err := outputShapeForNode(concatNode, "concat", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 8, 19456})
		})
	})
}

func TestPackedSwiGLUOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a packed swiglu node with one float input", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 8, 19456})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 8, 9728})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("gate_up", "input", ir.OpInput, inputShape)
		swigluNode := manifestComputeNode("swiglu_0", "activation.swiglu", ir.OpFused, outputShape)
		swigluNode.AddInput(inputNode)
		setFloat32ValueType(inputNode)
		setFloat32ValueType(swigluNode)

		input, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("gate_up", input)

		convey.Convey("It should halve the packed final dimension", func() {
			shape, err := outputShapeForNode(swigluNode, "swiglu", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 8, 9728})
		})
	})
}

func TestModulatedLayerNormOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a modulated layernorm node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 4096, 3072})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		modulationShape, err := tensor.NewShape([]int{1, 18432})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("h_0", "input", ir.OpInput, inputShape)
		modulationNode := manifestComputeNode("mod", "input", ir.OpInput, modulationShape)
		normNode := manifestComputeNode("norm", "math.modulated_layernorm", ir.OpFused, staleShape)
		normNode.AddInput(inputNode)
		normNode.AddInput(modulationNode)
		setFloat32ValueType(inputNode)
		setFloat32ValueType(modulationNode)
		setFloat32ValueType(normNode)

		input, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		modulation, err := tensor.NewZeroed(modulationShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("h_0", input)
		tensorWorkspace.Store("mod", modulation)

		convey.Convey("It should use the actual primary input shape", func() {
			shape, err := outputShapeForNode(normNode, "modulated_layernorm", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 4096, 3072})
		})
	})
}

func TestBatchNormDenormOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a batchnorm denorm node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 128, 64, 64})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 1, 1, 1})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("packed_latent", "input", ir.OpInput, inputShape)
		normNode := manifestComputeNode("bn", "math.batchnorm_denorm", ir.OpFused, staleShape)
		normNode.AddInput(inputNode)
		setFloat32ValueType(inputNode)
		setFloat32ValueType(normNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("packed_latent", input)

		convey.Convey("It should preserve the actual input shape", func() {
			shape, err := outputShapeForNode(normNode, "batchnorm_denorm", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 128, 64, 64})
		})
	})
}

func TestConv2DOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a conv2d node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 32, 128, 128})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 1, 1, 1})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("latent_image", "input", ir.OpInput, inputShape)
		convNode := manifestComputeNode("post_quant_conv", "convolution.conv2d", ir.OpFused, staleShape)
		convNode.AddInput(inputNode)
		convNode.SetAttribute("out_channels", ir.IntAttribute(32))
		convNode.SetAttribute("kernel_h", ir.IntAttribute(1))
		convNode.SetAttribute("kernel_w", ir.IntAttribute(1))
		convNode.SetAttribute("stride_h", ir.IntAttribute(1))
		convNode.SetAttribute("stride_w", ir.IntAttribute(1))
		convNode.SetAttribute("pad_h", ir.IntAttribute(0))
		convNode.SetAttribute("pad_w", ir.IntAttribute(0))
		setFloat32ValueType(inputNode)
		setFloat32ValueType(convNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("latent_image", input)

		convey.Convey("It should derive the convolution output shape", func() {
			shape, err := outputShapeForNode(convNode, "conv2d", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 32, 128, 128})
		})
	})
}

func TestUpsampleNearest2DOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given an upsample nearest2d node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 512, 128, 128})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 512, 128, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("up0_h2", "input", ir.OpInput, inputShape)
		upsampleNode := manifestComputeNode("up0_nearest", "shape.upsample_nearest2d", ir.OpFused, staleShape)
		upsampleNode.AddInput(inputNode)
		upsampleNode.SetAttribute("scale_h", ir.IntAttribute(2))
		upsampleNode.SetAttribute("scale_w", ir.IntAttribute(2))
		setFloat32ValueType(inputNode)
		setFloat32ValueType(upsampleNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("up0_h2", input)

		convey.Convey("It should scale the spatial dimensions", func() {
			shape, err := outputShapeForNode(upsampleNode, "upsample_nearest2d", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 512, 256, 256})
		})
	})
}

func TestMultiAxisRoPEOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a multi-axis RoPE node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 5120, 24, 128})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 1, 24, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("q", "input", ir.OpInput, inputShape)
		ropeNode := manifestComputeNode("tb_rope_q_0", "positional.multi_axis_rope", ir.OpFused, staleShape)
		ropeNode.AddInput(inputNode)
		setFloat32ValueType(inputNode)
		setFloat32ValueType(ropeNode)

		input, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("q", input)

		convey.Convey("It should use the actual primary input shape", func() {
			shape, err := outputShapeForNode(ropeNode, "multi_axis_rope", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 5120, 24, 128})
		})
	})
}

func TestSliceOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a slice node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 5120, 3072})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 1, 3072})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("joint", "input", ir.OpInput, inputShape)
		sliceNode := manifestComputeNode("context_slice", "shape.slice", ir.OpFused, staleShape)
		sliceNode.AddInput(inputNode)
		sliceNode.SetAttribute("dim", ir.IntAttribute(1))
		sliceNode.SetAttribute("start", ir.IntAttribute(0))
		sliceNode.SetAttribute("end", ir.IntAttribute(1024))
		setFloat32ValueType(inputNode)
		setFloat32ValueType(sliceNode)

		input, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("joint", input)

		convey.Convey("It should use the bounded slice length", func() {
			shape, err := outputShapeForNode(sliceNode, "slice", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 1024, 3072})
		})
	})
}

func TestTransposeOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a transpose node with runtime input shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 64, 64, 128})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 64, 64, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("packed_grid", "input", ir.OpInput, inputShape)
		transposeNode := manifestComputeNode("vae.unpack.pack_t23", "shape.transpose", ir.OpFused, staleShape)
		transposeNode.AddInput(inputNode)
		transposeNode.SetAttribute("dim0", ir.IntAttribute(2))
		transposeNode.SetAttribute("dim1", ir.IntAttribute(3))
		setFloat32ValueType(inputNode)
		setFloat32ValueType(transposeNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("packed_grid", input)

		convey.Convey("It should swap the requested dimensions", func() {
			shape, err := outputShapeForNode(transposeNode, "transpose", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 64, 128, 64})
		})
	})
}

func TestReshapeOutputShapeForNode(testingObject *testing.T) {
	convey.Convey("Given a reshape node with manifest target shape", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 4096, 128})
		convey.So(err, convey.ShouldBeNil)

		staleShape, err := tensor.NewShape([]int{1, 4096, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("latents", "input", ir.OpInput, inputShape)
		reshapeNode := manifestComputeNode("vae.unpack.grid", "shape.reshape", ir.OpFused, staleShape)
		reshapeNode.AddInput(inputNode)
		reshapeNode.SetAttribute("shape", ir.StringAttribute("[1,64,64,128]"))
		setFloat32ValueType(inputNode)
		setFloat32ValueType(reshapeNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("latents", input)

		convey.Convey("It should allocate the manifest target shape", func() {
			shape, err := outputShapeForNode(reshapeNode, "reshape", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 64, 64, 128})
		})
	})
}

func TestReshapeOutputShapeForLoweredListAttribute(testingObject *testing.T) {
	convey.Convey("Given a reshape node with a lowered YAML list attribute", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 4096, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("latents", "input", ir.OpInput, inputShape)
		reshapeNode := manifestComputeNode("vae.unpack.grid", "shape.reshape", ir.OpFused, inputShape)
		reshapeNode.AddInput(inputNode)
		reshapeNode.SetAttribute("shape", ir.StringAttribute("[1 64 64 128]"))

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("latents", input)

		convey.Convey("It should parse the whitespace separated dimensions", func() {
			shape, err := outputShapeForNode(reshapeNode, "reshape", tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Dims(), convey.ShouldResemble, []int{1, 64, 64, 128})
		})
	})
}

func TestNodeStorageDTypeUsesRuntimeInputTensor(testingObject *testing.T) {
	convey.Convey("Given a shape op whose input is a graph input node", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1, 4096, 128})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("latents", "input", ir.OpInput, inputShape)
		reshapeNode := manifestComputeNode("vae.unpack.grid", "shape.reshape", ir.OpFused, inputShape)
		reshapeNode.AddInput(inputNode)

		input, err := tensor.NewZeroed(inputShape, dtype.BFloat16)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("latents", input)

		convey.Convey("It should preserve the resident input dtype", func() {
			storageDType := nodeStorageDType(reshapeNode, tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(storageDType, convey.ShouldEqual, dtype.BFloat16)
		})
	})
}

func TestNodeStorageDTypeUsesValueTypeForTimestep(testingObject *testing.T) {
	convey.Convey("Given a timestep embedding fed by an F32 scalar input", testingObject, func() {
		inputShape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		outputShape, err := tensor.NewShape([]int{1, 256})
		convey.So(err, convey.ShouldBeNil)

		inputNode := manifestComputeNode("timestep", "input", ir.OpInput, inputShape)
		timestepNode := manifestComputeNode("time_proj", "embedding.timestep", ir.OpFused, outputShape)
		timestepNode.AddInput(inputNode)
		setFloat32ValueType(inputNode)
		timestepNode.SetValueType(ir.ValueType{DType: dtype.BFloat16})

		input, err := tensor.NewZeroed(inputShape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		tensorWorkspace := newWorkspace()
		defer tensorWorkspace.Close()
		tensorWorkspace.Store("timestep", input)

		convey.Convey("It should use the lowered output dtype", func() {
			storageDType := nodeStorageDType(timestepNode, tensorWorkspace, "", nil, newManifestBindings(nil))

			convey.So(storageDType, convey.ShouldEqual, dtype.BFloat16)
		})
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
