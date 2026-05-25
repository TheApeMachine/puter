package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
)

/*
executionDevice is the minimal slice of device.Backend the dispatcher
actually invokes. Narrowing the dependency keeps tests honest (a mock
only has to satisfy the methods the dispatcher calls) and makes it easy
to audit what surface the executor depends on.

Every device.Backend implementation (cpu.Backend, metal.Backend,
cuda.Backend, xla.Backend) satisfies this interface implicitly because
they all embed the same family sub-interfaces from device/interface.go.
*/
type executionDevice interface {
	// Embedding family.
	Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType)

	// LayerNorm family.
	RMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType)
	LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType)

	// Matmul family.
	Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType)

	// Elementwise family (subset).
	Add(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Div(dst, left, right unsafe.Pointer, count int, format dtype.DType)

	// Activation family (subset).
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	Tanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	Gelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	Silu(dst, src unsafe.Pointer, count int, format dtype.DType)
}

/*
dispatcher walks one ExecutionPlan against an ast.Graph and invokes the
device backend for each node. It owns a per-call valueTable and shares
the host memory backend, device backend, and weight store with the
parent execution.Backend.
*/
type dispatcher struct {
	values        *valueTable
	graph         *ast.Graph
	plan          *runtime.ExecutionPlan
	nodeByID      map[string]*ast.GraphNode
	deviceBackend executionDevice
	memory        tensor.Backend
	weights       WeightStore
}

func newDispatcher(
	graph *ast.Graph,
	plan *runtime.ExecutionPlan,
	deviceBackend executionDevice,
	memory tensor.Backend,
	weights WeightStore,
) *dispatcher {
	nodeByID := make(map[string]*ast.GraphNode, len(graph.Nodes))

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		nodeByID[node.ID] = node
	}

	return &dispatcher{
		values:        newValueTable(),
		graph:         graph,
		plan:          plan,
		nodeByID:      nodeByID,
		deviceBackend: deviceBackend,
		memory:        memory,
		weights:       weights,
	}
}

/*
run walks every node in the plan's layers and dispatches it. The dispatcher
is sequential today; layer parallelism lands once the async DAG executor
is in place (ARCHITECTURE.md §5.2). The contract is preserved: nodes
within one layer are independent and could run concurrently.
*/
func (dispatcher *dispatcher) run() error {
	for layerIndex, layer := range dispatcher.plan.Layers {
		for _, nodeID := range layer {
			node, ok := dispatcher.nodeByID[nodeID]

			if !ok {
				return fmt.Errorf(
					"execution: plan layer %d references unknown node %q",
					layerIndex, nodeID,
				)
			}

			if err := dispatcher.runNode(node); err != nil {
				return fmt.Errorf("execution: node %q (%s): %w", node.ID, node.Op, err)
			}
		}
	}

	return nil
}

/*
runNode picks one of three execution paths: fused (kernel attached by
codegen), known device op (table-driven dispatch into device.Backend), or
an explicit unsupported error so missing ops are surfaced cleanly.
*/
func (dispatcher *dispatcher) runNode(node *ast.GraphNode) error {
	if node.Op == optimizer.FuseOp {
		return dispatcher.runFusedNode(node)
	}

	handler, ok := opTable[node.Op]

	if !ok {
		return fmt.Errorf("unsupported op %q (no dispatcher registered)", node.Op)
	}

	return handler(dispatcher, node)
}

/*
runFusedNode picks the CPU kernel attached by codegen and runs it directly.
Metal-resident fusion lands once metal.Backend exposes a host-visible
view; this path currently fails over to "no CPU kernel attached" if the
graph was compiled with TargetMetal only.
*/
func (dispatcher *dispatcher) runFusedNode(node *ast.GraphNode) error {
	setAny, ok := node.Attributes[codegen.KernelAttribute]

	if !ok {
		return fmt.Errorf("fused node missing %q attribute", codegen.KernelAttribute)
	}

	kernelSet, ok := setAny.(*codegen.KernelSet)

	if !ok {
		return fmt.Errorf(
			"fused node %q attribute is %T, want *codegen.KernelSet",
			codegen.KernelAttribute, setAny,
		)
	}

	cpuAny := kernelSet.For(codegen.TargetCPU)

	if cpuAny == nil {
		return fmt.Errorf("fused node has no TargetCPU kernel attached")
	}

	cpuKernel, ok := cpuAny.(*codegen.CPUKernel)

	if !ok {
		return fmt.Errorf("kernel for TargetCPU is %T, want *codegen.CPUKernel", cpuAny)
	}

	inputBuffers := make([][]float32, 0, len(cpuKernel.Inputs()))
	var count int

	for _, inputName := range cpuKernel.Inputs() {
		inputTensor, err := dispatcher.values.tensor(inputName)

		if err != nil {
			return err
		}

		values, err := inputTensor.Float32Native()

		if err != nil {
			return fmt.Errorf("fused node input %q: %w", inputName, err)
		}

		inputBuffers = append(inputBuffers, values)

		if count == 0 {
			count = len(values)
		}
	}

	outputTensor, err := dispatcher.allocateLike(inputBuffers[0], count)

	if err != nil {
		return err
	}

	outputBuffer, err := outputTensor.Float32Native()

	if err != nil {
		return fmt.Errorf("fused node output allocation: %w", err)
	}

	if err := cpuKernel.Run(inputBuffers, outputBuffer, count); err != nil {
		return err
	}

	dispatcher.values.set(node.ID, outputTensor)

	return nil
}

/*
allocateLike returns a new host tensor of the requested length, initialised
to zero. Used both by the fused kernel runner and by the elementwise
shape-preserving op handlers in dispatch_table.go.
*/
func (dispatcher *dispatcher) allocateLike(reference []float32, count int) (tensor.Tensor, error) {
	_ = reference

	shape, err := tensor.NewShape([]int{count})

	if err != nil {
		return nil, fmt.Errorf("execution: derive shape: %w", err)
	}

	byteCount, err := dtype.Float32.BytesFor(count)

	if err != nil {
		return nil, fmt.Errorf("execution: derive byte count: %w", err)
	}

	return dispatcher.memory.Upload(shape, dtype.Float32, make([]byte, byteCount))
}

/*
pointerOf returns an unsafe.Pointer into the first element of one tensor's
Float32 storage. Used to bridge tensor.Tensor → device.Backend's
unsafe.Pointer contract. Callers must ensure the tensor remains live for
the duration of the device call (the dispatcher holds it through the
valueTable for the rest of the graph call).
*/
func pointerOf(input tensor.Tensor) (unsafe.Pointer, int, error) {
	if input == nil {
		return nil, 0, fmt.Errorf("execution: tensor is required")
	}

	storage, err := input.Float32Native()

	if err != nil {
		return nil, 0, err
	}

	if len(storage) == 0 {
		return nil, 0, nil
	}

	return unsafe.Pointer(&storage[0]), len(storage), nil
}
