package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
opHandler runs one node of a known op kind. Each handler reads its inputs
and config from the dispatcher's value table, allocates an output tensor
via dispatcher.memory, calls into device.Backend, and writes the output
back into the value table under node.ID.
*/
type opHandler func(dispatcher *dispatcher, node *ast.GraphNode) error

/*
opTable maps ast.GraphNode.Op strings to their device.Backend handlers.
Adding a new op means writing a handler here — not editing device.Backend
or any kernel. The list intentionally covers only ops a Llama-style chat
model and a simple diffusion denoiser actually hit; broader coverage
(every op declared under template/operation/) lands as additional models
exercise it.
*/
var opTable = map[string]opHandler{
	"embedding.token":    handleEmbeddingToken,
	"math.rmsnorm":       handleRMSNorm,
	"math.layernorm":     handleLayerNorm,
	"math.matmul":        handleMatmul,
	"projection.linear":  handleMatmul,
	"math.add":           handleBinaryElementwise(opAdd),
	"math.sub":           handleBinaryElementwise(opSub),
	"math.mul":           handleBinaryElementwise(opMul),
	"math.div":           handleBinaryElementwise(opDiv),
	"activation.relu":    handleUnaryActivation(activationReLU),
	"activation.sigmoid": handleUnaryActivation(activationSigmoid),
	"activation.tanh":    handleUnaryActivation(activationTanh),
	"activation.gelu":    handleUnaryActivation(activationGelu),
	"activation.swish":   handleUnaryActivation(activationSilu),
	"value.assign":       handleAssign,
}

/*
handleAssign forwards one input value to the node's output unchanged.
Useful for graph-level naming and for the value-identity pattern.
*/
func handleAssign(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("value.assign expects exactly one input, got %d", len(node.Inputs))
	}

	value, ok := dispatcher.values.get(node.Inputs[0])

	if !ok {
		return fmt.Errorf("value.assign input %q not found", node.Inputs[0])
	}

	dispatcher.values.set(node.ID, value)

	return nil
}

/*
handleEmbeddingToken implements the embedding lookup op. The table tensor
comes from the weight store (node.Weights.TensorName); the index tensor
comes from the value table (node.Inputs[0]).
*/
func handleEmbeddingToken(dispatcher *dispatcher, node *ast.GraphNode) error {
	if node.Weights == nil || node.Weights.TensorName == "" {
		return fmt.Errorf("embedding.token requires Weights.TensorName")
	}

	if len(node.Inputs) != 1 {
		return fmt.Errorf("embedding.token expects one input, got %d", len(node.Inputs))
	}

	table, err := dispatcher.weights.Lookup(node.Weights.TensorName)

	if err != nil {
		return fmt.Errorf("embedding.token weight %q: %w", node.Weights.TensorName, err)
	}

	indices, err := dispatcher.values.tokenIDs(node.Inputs[0])

	if err != nil {
		return err
	}

	vocab := intAttr(node, "vocab_size", 0)
	hidden := intAttr(node, "d_model", 0)

	if vocab == 0 || hidden == 0 {
		tableShape := table.Shape().Dims()

		if len(tableShape) >= 2 {
			vocab = tableShape[0]
			hidden = tableShape[1]
		}
	}

	if vocab == 0 || hidden == 0 {
		return fmt.Errorf("embedding.token cannot derive vocab/hidden dims")
	}

	// Pack the host-side token IDs into an Int32 byte buffer and upload it
	// through the active memory backend so the kernel sees a resident
	// tensor handle rather than a Go heap pointer. ARCHITECTURE.md §7
	// bans passing Go slice headers to Metal/CUDA/XLA — the dispatcher
	// can't predict which backend will run, so we always upload. CPU
	// (HostBackend) returns a HostTensor that aliases the upload bytes;
	// Metal returns a DeviceTensor over an MTLBuffer.
	indexBytes := make([]byte, len(indices)*4)

	for index, value := range indices {
		bytePointer := unsafe.Pointer(&indexBytes[index*4])
		*(*int32)(bytePointer) = int32(value)
	}

	indicesShape, err := tensor.NewShape([]int{len(indices)})

	if err != nil {
		return err
	}

	indicesTensor, err := dispatcher.memory.Upload(indicesShape, dtype.Int32, indexBytes)

	if err != nil {
		return err
	}

	outputShape, err := tensor.NewShape([]int{len(indices), hidden})

	if err != nil {
		return err
	}

	outputBytes, err := dtype.Float32.BytesFor(len(indices) * hidden)

	if err != nil {
		return err
	}

	output, err := dispatcher.allocateOutput(node, outputShape, dtype.Float32, outputBytes)

	if err != nil {
		return err
	}

	tablePointer, _, err := pointerOf(table)

	if err != nil {
		return err
	}

	indicesPointer, _, err := pointerOf(indicesTensor)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.Lookup(
		tablePointer, indicesPointer, outputPointer,
		vocab, hidden, len(indices),
		dtype.Float32,
	)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleRMSNorm implements math.rmsnorm. Inputs: the activation tensor; the
scale weight (from node.Weights). Config: eps and last_dim are read from
node.Attributes.
*/
func handleRMSNorm(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("math.rmsnorm expects one input, got %d", len(node.Inputs))
	}

	if node.Weights == nil || node.Weights.TensorName == "" {
		return fmt.Errorf("math.rmsnorm requires Weights.TensorName")
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	scale, err := dispatcher.weights.Lookup(node.Weights.TensorName)

	if err != nil {
		return fmt.Errorf("math.rmsnorm weight %q: %w", node.Weights.TensorName, err)
	}

	rows, lastDim := matrixDims(input)

	output, err := dispatcher.allocateOutput(node, input.Shape(), dtype.Float32, rows*lastDim*4)

	if err != nil {
		return err
	}

	inputPtr, _, err := pointerOf(input)

	if err != nil {
		return err
	}

	scalePtr, _, err := pointerOf(scale)

	if err != nil {
		return err
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.RMSNorm(
		inputPtr, scalePtr, outputPtr,
		rows, lastDim,
		dtype.Float32,
	)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleLayerNorm implements math.layernorm. Same shape as RMSNorm but with
both scale and bias weights.
*/
func handleLayerNorm(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) != 1 {
		return fmt.Errorf("math.layernorm expects one input, got %d", len(node.Inputs))
	}

	input, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	scaleName := ""
	biasName := ""

	if node.Weights != nil {
		scaleName = node.Weights.TensorName
	}

	if scaleName == "" {
		return fmt.Errorf("math.layernorm requires a scale weight")
	}

	scale, err := dispatcher.weights.Lookup(scaleName)

	if err != nil {
		return fmt.Errorf("math.layernorm scale %q: %w", scaleName, err)
	}

	var bias tensor.Tensor

	if biasName != "" {
		bias, err = dispatcher.weights.Lookup(biasName)

		if err != nil {
			return fmt.Errorf("math.layernorm bias %q: %w", biasName, err)
		}
	}

	rows, lastDim := matrixDims(input)

	output, err := dispatcher.allocateOutput(node, input.Shape(), dtype.Float32, rows*lastDim*4)

	if err != nil {
		return err
	}

	inputPtr, _, err := pointerOf(input)

	if err != nil {
		return err
	}

	scalePtr, _, err := pointerOf(scale)

	if err != nil {
		return err
	}

	var biasPtr unsafe.Pointer

	if bias != nil {
		biasPtr, _, err = pointerOf(bias)

		if err != nil {
			return err
		}
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.LayerNorm(
		inputPtr, scalePtr, biasPtr, outputPtr,
		rows, lastDim,
		dtype.Float32,
	)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
handleMatmul implements math.matmul and projection.linear. For
projection.linear the weight tensor is fetched from the weight store;
for raw math.matmul the second operand is read from the value table.
*/
func handleMatmul(dispatcher *dispatcher, node *ast.GraphNode) error {
	if len(node.Inputs) < 1 {
		return fmt.Errorf("%s expects at least one input", node.Op)
	}

	left, err := dispatcher.values.tensor(node.Inputs[0])

	if err != nil {
		return err
	}

	var right tensor.Tensor

	if node.Op == "projection.linear" {
		if node.Weights == nil || node.Weights.TensorName == "" {
			return fmt.Errorf("projection.linear requires Weights.TensorName")
		}

		// HuggingFace stores nn.Linear weights as [out_features, in_features]
		// and the canonical forward pass computes y = x @ W.T. The generic
		// device.Matmul kernel expects row-major [inner, cols], so we ask
		// the weight store for the transposed handle when it offers one.
		// Without this every projection.linear would surface as a shape
		// mismatch the moment the dispatcher runs against real Llama-style
		// weights (e.g. k_proj's [512, 2048] vs the post-norm hidden state
		// [seq, 2048]).
		if transposed, ok := dispatcher.weights.(TransposedLookup); ok {
			right, err = transposed.LookupTransposed(node.Weights.TensorName)
		} else {
			return fmt.Errorf(
				"projection.linear weight %q: weight store does not implement TransposedLookup; HuggingFace Linear weights need a transposed view (see puter/execution.TransposedLookup)",
				node.Weights.TensorName,
			)
		}

		if err != nil {
			return fmt.Errorf("projection.linear weight %q: %w", node.Weights.TensorName, err)
		}
	} else {
		if len(node.Inputs) < 2 {
			return fmt.Errorf("math.matmul expects two inputs, got %d", len(node.Inputs))
		}

		right, err = dispatcher.values.tensor(node.Inputs[1])

		if err != nil {
			return err
		}
	}

	leftDims := left.Shape().Dims()
	rightDims := right.Shape().Dims()

	if len(leftDims) < 2 || len(rightDims) < 2 {
		return fmt.Errorf("matmul requires rank-2 operands, got %v × %v", leftDims, rightDims)
	}

	rows := productOf(leftDims[:len(leftDims)-1])
	inner := leftDims[len(leftDims)-1]
	cols := rightDims[len(rightDims)-1]

	if inner != rightDims[len(rightDims)-2] {
		return fmt.Errorf("matmul inner dim mismatch: left %v, right %v", leftDims, rightDims)
	}

	outputShape, err := tensor.NewShape(append(append([]int(nil), leftDims[:len(leftDims)-1]...), cols))

	if err != nil {
		return err
	}

	output, err := dispatcher.allocateOutput(node, outputShape, dtype.Float32, rows*cols*4)

	if err != nil {
		return err
	}

	leftPtr, _, err := pointerOf(left)

	if err != nil {
		return err
	}

	rightPtr, _, err := pointerOf(right)

	if err != nil {
		return err
	}

	outputPtr, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	dispatcher.deviceBackend.Matmul(
		outputPtr, leftPtr, rightPtr,
		rows, inner, cols,
		dtype.Float32,
	)

	dispatcher.values.set(node.ID, output)

	return nil
}

/*
binaryOp identifies the device.Backend Elementwise method to call.
*/
type binaryOp int

const (
	opAdd binaryOp = iota
	opSub
	opMul
	opDiv
)

func handleBinaryElementwise(operation binaryOp) opHandler {
	return func(dispatcher *dispatcher, node *ast.GraphNode) error {
		if len(node.Inputs) != 2 {
			return fmt.Errorf("%s expects two inputs, got %d", node.Op, len(node.Inputs))
		}

		left, err := dispatcher.values.tensor(node.Inputs[0])

		if err != nil {
			return err
		}

		right, err := dispatcher.values.tensor(node.Inputs[1])

		if err != nil {
			return err
		}

		count := left.Len()

		output, err := dispatcher.allocateOutput(node, left.Shape(), dtype.Float32, count*4)

		if err != nil {
			return err
		}

		leftPtr, _, err := pointerOf(left)

		if err != nil {
			return err
		}

		rightPtr, _, err := pointerOf(right)

		if err != nil {
			return err
		}

		outputPtr, _, err := pointerOf(output)

		if err != nil {
			return err
		}

		switch operation {
		case opAdd:
			dispatcher.deviceBackend.Add(outputPtr, leftPtr, rightPtr, count, dtype.Float32)
		case opSub:
			dispatcher.deviceBackend.Sub(outputPtr, leftPtr, rightPtr, count, dtype.Float32)
		case opMul:
			dispatcher.deviceBackend.Mul(outputPtr, leftPtr, rightPtr, count, dtype.Float32)
		case opDiv:
			dispatcher.deviceBackend.Div(outputPtr, leftPtr, rightPtr, count, dtype.Float32)
		}

		dispatcher.values.set(node.ID, output)

		return nil
	}
}

/*
unaryActivation identifies the device.Backend Activation method to call.
*/
type unaryActivation int

const (
	activationReLU unaryActivation = iota
	activationSigmoid
	activationTanh
	activationGelu
	activationSilu
)

func handleUnaryActivation(activation unaryActivation) opHandler {
	return func(dispatcher *dispatcher, node *ast.GraphNode) error {
		if len(node.Inputs) != 1 {
			return fmt.Errorf("%s expects one input, got %d", node.Op, len(node.Inputs))
		}

		input, err := dispatcher.values.tensor(node.Inputs[0])

		if err != nil {
			return err
		}

		count := input.Len()

		output, err := dispatcher.allocateOutput(node, input.Shape(), dtype.Float32, count*4)

		if err != nil {
			return err
		}

		inputPtr, _, err := pointerOf(input)

		if err != nil {
			return err
		}

		outputPtr, _, err := pointerOf(output)

		if err != nil {
			return err
		}

		switch activation {
		case activationReLU:
			dispatcher.deviceBackend.ReLU(outputPtr, inputPtr, count, dtype.Float32)
		case activationSigmoid:
			dispatcher.deviceBackend.Sigmoid(outputPtr, inputPtr, count, dtype.Float32)
		case activationTanh:
			dispatcher.deviceBackend.Tanh(outputPtr, inputPtr, count, dtype.Float32)
		case activationGelu:
			dispatcher.deviceBackend.Gelu(outputPtr, inputPtr, count, dtype.Float32)
		case activationSilu:
			dispatcher.deviceBackend.Silu(outputPtr, inputPtr, count, dtype.Float32)
		}

		dispatcher.values.set(node.ID, output)

		return nil
	}
}

/*
matrixDims collapses every leading dimension into "rows" and treats the
last dimension as the per-row feature width. Standard convention for the
RMSNorm/LayerNorm device ops.
*/
func matrixDims(input tensor.Tensor) (rows int, lastDim int) {
	dims := input.Shape().Dims()

	if len(dims) == 0 {
		return 0, 0
	}

	lastDim = dims[len(dims)-1]
	rows = productOf(dims[:len(dims)-1])

	if rows == 0 {
		rows = 1
	}

	return rows, lastDim
}

func productOf(dims []int) int {
	product := 1

	for _, dimension := range dims {
		product *= dimension
	}

	return product
}

func zeroBytes(count int) []byte {
	if count <= 0 {
		return nil
	}

	return make([]byte, count*4)
}

func intAttr(node *ast.GraphNode, key string, fallback int) int {
	if node == nil || node.Attributes == nil {
		return fallback
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return fallback
	}

	switch typed := raw.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return fallback
	}
}
