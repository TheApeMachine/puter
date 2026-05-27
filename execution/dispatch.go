package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
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
	TimestepEmbedding(
		config device.TimestepEmbeddingConfig,
		timesteps, output unsafe.Pointer,
		count, dim int,
		format dtype.DType,
	)

	// LayerNorm family.
	RMSNorm(config device.RMSNormConfig, input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType)
	AdaptiveRMSNorm(
		config device.RMSNormConfig,
		input, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols int,
		format dtype.DType,
	)
	LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType)
	GroupNorm(
		config device.GroupNormConfig,
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	ModulatedLayerNorm(
		config device.ModulatedLayerNormConfig,
		input, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols int,
		format dtype.DType,
	)

	// Matmul family.
	Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType)

	// Convolution family.
	Conv2D(
		config device.Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)

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
	SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)

	// RoPE family. Applies rotary position embeddings in place on the
	// per-head query/key tensors. Config carries the base frequency and
	// the starting absolute position so KV-cache extends correctly.
	RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType)
	MultiAxisRoPE(
		config device.MultiAxisRoPEConfig,
		input, output unsafe.Pointer,
		batch, seqLen, numHeads, headDim int,
		format dtype.DType,
	)

	// Attention family.
	MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
}

type batchExecutionDevice interface {
	BeginBatch() error
	EndBatch() error
}

/*
dispatcher walks one ExecutionPlan against an ast.Graph and invokes the
device backend for each node. It owns a per-call valueTable and shares
the host memory backend, device backend, and weight store with the
parent execution.Backend.
*/
type dispatcher struct {
	values         *valueTable
	graph          *ast.Graph
	graphName      string
	plan           *runtime.ExecutionPlan
	nodeByID       map[string]*ast.GraphNode
	deviceBackend  executionDevice
	memory         tensor.Backend
	weights        WeightStore
	workspaces     *WorkspaceMap
	maxBindings    ir.SymbolMap
	launchBindings ir.SymbolMap
}

func newDispatcher(
	graphName string,
	graph *ast.Graph,
	plan *runtime.ExecutionPlan,
	deviceBackend executionDevice,
	memory tensor.Backend,
	weights WeightStore,
	workspaces *WorkspaceMap,
	launchBindings ir.SymbolMap,
) *dispatcher {
	nodeByID := make(map[string]*ast.GraphNode, len(graph.Nodes))

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		nodeByID[node.ID] = node
	}

	dispatcher := &dispatcher{
		values:         newValueTable(),
		graph:          graph,
		graphName:      graphName,
		plan:           plan,
		nodeByID:       nodeByID,
		deviceBackend:  deviceBackend,
		memory:         memory,
		weights:        weights,
		workspaces:     workspaces,
		launchBindings: launchBindings,
	}

	if workspaces != nil {
		dispatcher.maxBindings = workspaces.MaxBindings(graphName)
	}

	return dispatcher
}

/*
workspaceOutput returns the pre-planned workspace tensor for one node's
output, or nil when the graph has no attached workspace for this node
(e.g., a graph the planner couldn't size, or a backend that bypasses
the workspace path entirely). Handlers call this in place of
dispatcher.memory.Upload(...) for output allocation; when it returns a
tensor the handler writes through that tensor's DispatchPointer.

The lookup is by graph name + node ID. Both are known at dispatcher
construction time, so this is a flat map probe — no allocation, no
contention.
*/
func (dispatcher *dispatcher) workspaceOutput(nodeID string) tensor.Tensor {
	if dispatcher.workspaces == nil {
		return nil
	}

	t, ok := dispatcher.workspaces.OutputFor(dispatcher.graphName, nodeID)

	if !ok {
		return nil
	}

	return t
}

/*
allocateOutput is the single entry point every handler uses to acquire
its output tensor. When the backend has an attached workspace for this
graph, the planner-allocated tensor is returned with zero per-call
allocation. When there is no workspace (unit tests of the dispatcher,
or backends that bypass the planner entirely), the helper falls back
to dispatcher.memory.Upload(...) for the legacy per-call allocation
path so those callers continue to work.

shape, dataType, and byteCount must agree (shape.Bytes(dataType) ==
byteCount). The fallback path verifies through Upload; the workspace
path trusts the planner produced a correctly-sized region.
*/
func (dispatcher *dispatcher) allocateOutput(
	node *ast.GraphNode,
	shape tensor.Shape,
	dataType dtype.DType,
	byteCount int,
) (tensor.Tensor, error) {
	if output := dispatcher.workspaceOutput(node.ID); output != nil {
		return output, nil
	}

	return dispatcher.memory.Upload(shape, dataType, make([]byte, byteCount))
}

/*
run walks every node in the plan's layers and dispatches it. The dispatcher
is sequential today; layer parallelism lands once the async DAG executor
is in place (ARCHITECTURE.md §5.2). The contract is preserved: nodes
within one layer are independent and could run concurrently.
*/
func (dispatcher *dispatcher) run() (err error) {
	batcher, batching, err := dispatcher.beginBatch()

	if err != nil {
		return err
	}

	if batching {
		defer func() {
			endErr := batcher.EndBatch()

			if err == nil && endErr != nil {
				err = endErr
			}
		}()
	}

	return dispatcher.runLayers()
}

func (dispatcher *dispatcher) beginBatch() (batchExecutionDevice, bool, error) {
	if !dispatcher.canBatchDevice() {
		return nil, false, nil
	}

	batcher := dispatcher.deviceBackend.(batchExecutionDevice)

	if err := batcher.BeginBatch(); err != nil {
		return nil, false, err
	}

	return batcher, true, nil
}

func (dispatcher *dispatcher) canBatchDevice() bool {
	if _, ok := dispatcher.deviceBackend.(batchExecutionDevice); !ok {
		return false
	}

	for _, node := range dispatcher.graph.Nodes {
		if node == nil {
			continue
		}

		if dispatcher.nodeReadsDeviceScalar(node) {
			return false
		}
	}

	return true
}

func (dispatcher *dispatcher) nodeReadsDeviceScalar(node *ast.GraphNode) bool {
	return node.Op == "positional.rope" && len(node.Inputs) >= 2
}

func (dispatcher *dispatcher) runLayers() error {
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

	if node.Op == "value.assign" {
		return dispatcher.runAssign(node)
	}

	registry, err := defaultOperationRegistry()

	if err != nil {
		return err
	}

	bind, err := registry.Bind(node)

	if err != nil {
		return err
	}

	return runBoundNode(dispatcher, node, bind)
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
to zero. Used by the fused kernel runner.
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

func (dispatcher *dispatcher) runAssign(node *ast.GraphNode) error {
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
DispatchPointer is the optional interface a tensor.Tensor implementation
advertises when it can produce the unsafe.Pointer the active device.Backend
expects. Host-resident tensors return a pointer into their byte storage;
device-resident tensors return a pointer to the device-tensor struct itself,
which each backend's bridge unwraps (see puter/device/metal's
resolveDeviceTensor / resolveBufferRef pair).

This is the bridge that lets one dispatcher dispatch into any backend
without hard-coding tensor types. Implementations that don't advertise it
fall back to the legacy Float32Native path below.
*/
type DispatchPointer interface {
	DispatchPointer() unsafe.Pointer
}

/*
pointerOf returns an unsafe.Pointer suitable for handing to the active
device.Backend's kernels. For tensors that implement DispatchPointer
(host buffers, Metal DeviceTensor, future CUDA/XLA device tensors) the
returned pointer is whatever that backend's bridge expects. For tensors
that don't, we fall back to Float32Native — the legacy behaviour kept so
mock tensors in tests keep working.

The second return value is the element count, derived from Tensor.Len(),
so callers can inspect length without an extra interface call.

Callers must keep the originating tensor live for the duration of the
device call. The dispatcher's valueTable retains every tensor it produces
until the graph call returns, so this is enforced naturally.
*/
func pointerOf(input tensor.Tensor) (unsafe.Pointer, int, error) {
	if input == nil {
		return nil, 0, fmt.Errorf("execution: tensor is required")
	}

	if dispatchable, ok := input.(DispatchPointer); ok {
		return dispatchable.DispatchPointer(), input.Len(), nil
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
