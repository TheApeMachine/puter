package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Workspace owns a single contiguous host-resident byte region whose
layout was produced by the manifesto planner (ir.PlanWorkspace). It
hands out tensor.Tensor handles aliased to specific port allocations
so the dispatcher can replace per-call Upload calls with direct
references to the pre-planned region.

Backing storage comes from manifesto/tensor.Allocate, which returns
slab-allocated (small) or mmap-backed (medium/large) memory — both
guaranteed 64-byte aligned and outside the Go heap per
ARCHITECTURE.md §5.2's GC-safety requirement.

A Workspace is scoped to one compiled graph. The runtime executor
constructs one per graph at session init via Backend.AttachWorkspace,
keeps it alive for the session's duration, and Close()s it when the
session tears down.
*/
type Workspace struct {
	storage []byte
	layout  ir.WorkspaceLayout
	tensors map[int32]tensor.Tensor
}

/*
NewWorkspace allocates the planned workspace region from the manifesto
tensor allocator and pre-builds tensor handles for every port that
carries an allocation in the supplied *ir.Topology. The topology must
have been processed by PlanWorkspace; otherwise Allocations is empty
and the returned workspace will simply be unusable, which surfaces at
TensorFor time rather than silently.
*/
func NewWorkspace(topology *ir.Topology) (*Workspace, error) {
	if topology == nil {
		return nil, fmt.Errorf("execution: workspace topology is required")
	}

	layout := topology.Workspace

	if layout.Size <= 0 {
		return nil, fmt.Errorf(
			"execution: workspace layout has zero size — was PlanWorkspace run?",
		)
	}

	storage, err := tensor.Allocate(int(layout.Size))

	if err != nil {
		return nil, fmt.Errorf("execution: allocate workspace: %w", err)
	}

	workspace := &Workspace{
		storage: storage,
		layout:  layout,
		tensors: make(map[int32]tensor.Tensor, len(layout.Allocations)),
	}

	if err := workspace.preBuildTensors(topology); err != nil {
		_ = workspace.Close()

		return nil, err
	}

	return workspace, nil
}

/*
preBuildTensors walks every Port in the topology, slices the workspace
storage at the planned offset for that port, and constructs a HostTensor
aliasing the slice. The HostTensor uses the tensor package's workspace
sentinel so Close is a no-op — the Workspace itself owns the storage
lifetime.

Each port is keyed by Port.ID, which the planner assigned. The shared-
pointer invariant from TopologyForPlanning means producer and consumer
ports for one logical tensor have the same Port.ID, so the dispatcher
that walks the graph in either direction looks up the same tensor.
*/
func (workspace *Workspace) preBuildTensors(topology *ir.Topology) error {
	bindings := topology.Workspace.Allocations

	allocationByPortID := make(map[int32]ir.Interval, len(bindings))

	for _, interval := range bindings {
		allocationByPortID[interval.PortID] = interval
	}

	for _, node := range topology.Nodes {
		for _, port := range node.Outputs {
			if err := workspace.materializePort(port, allocationByPortID); err != nil {
				return err
			}
		}

		for _, port := range node.Inputs {
			if err := workspace.materializePort(port, allocationByPortID); err != nil {
				return err
			}
		}
	}

	return nil
}

func (workspace *Workspace) materializePort(
	port *ir.Port,
	allocationByPortID map[int32]ir.Interval,
) error {
	if port == nil || port.ID == 0 {
		return nil
	}

	if _, exists := workspace.tensors[port.ID]; exists {
		return nil
	}

	interval, ok := allocationByPortID[port.ID]

	if !ok {
		return fmt.Errorf(
			"execution: port %d has no allocation interval (planner output mismatch)",
			port.ID,
		)
	}

	end := interval.Offset + interval.Size

	if end > int64(len(workspace.storage)) {
		return fmt.Errorf(
			"execution: port %d allocation [%d, %d) exceeds workspace size %d",
			port.ID, interval.Offset, end, len(workspace.storage),
		)
	}

	// Three-index slice with capacity bounded so downstream code can't
	// accidentally extend into adjacent ports.
	aliased := workspace.storage[interval.Offset:end:end]

	shape, err := resolveShape(port.Type.ShapeSchema, port.Type.DType)

	if err != nil {
		return fmt.Errorf("execution: port %d shape: %w", port.ID, err)
	}

	workspace.tensors[port.ID] = tensor.NewAliasedHostTensor(shape, port.Type.DType, aliased)

	return nil
}

/*
TensorByPortID returns the pre-built tensor for one planned port. The
returned tensor aliases workspace storage; Close on the returned tensor
is a no-op (the workspace owns the lifetime). Returns an error when the
port ID has no entry — that means the planner did not allocate it,
which is a real bug the caller should surface.
*/
func (workspace *Workspace) TensorByPortID(portID int32) (tensor.Tensor, error) {
	if workspace == nil {
		return nil, fmt.Errorf("execution: nil workspace")
	}

	t, ok := workspace.tensors[portID]

	if !ok {
		return nil, fmt.Errorf("execution: port %d not allocated in workspace", portID)
	}

	return t, nil
}

/*
Layout returns the WorkspaceLayout the workspace was built against, for
diagnostics and for tests that need to assert on planner output.
*/
func (workspace *Workspace) Layout() ir.WorkspaceLayout {
	if workspace == nil {
		return ir.WorkspaceLayout{}
	}

	return workspace.layout
}

/*
Close releases the workspace's backing storage back to the manifesto
tensor allocator. Idempotent; calling Close on a workspace whose
tensors are still in use is undefined — the dispatcher must finish
its session before the workspace is closed.
*/
func (workspace *Workspace) Close() error {
	if workspace == nil || workspace.storage == nil {
		return nil
	}

	tensor.Release(workspace.storage)
	workspace.storage = nil
	workspace.tensors = nil

	return nil
}

/*
resolveShape converts an ir.ShapeSchema into a tensor.Shape. Symbolic
dimensions that were not resolved by the typer surface as errors here
rather than as silent zero-sized allocations — the planner should have
already bound them during AnalyzeLiveness (PortByteSize would have
errored on them), but defence in depth keeps the dispatcher honest.

Scalar (rank-0) shapes return an empty Shape, which downstream kernels
treat as a one-element tensor (matching the convention in
ir.PortByteSize).
*/
func resolveShape(schema ir.ShapeSchema, dataType dtype.DType) (tensor.Shape, error) {
	dims := make([]int, 0, len(schema.Dimensions))

	for index, dimension := range schema.Dimensions {
		if dimension.IsSymbolic() {
			return tensor.Shape{}, fmt.Errorf(
				"dim[%d] symbol %q unresolved at workspace materialization",
				index, dimension.Symbol,
			)
		}

		dims = append(dims, int(dimension.Static))
	}

	_ = dataType

	return tensor.NewShape(dims)
}

/*
WorkspaceMap indexes per-graph workspaces by name. Backend stores one
per CallGraph entrypoint so a multi-graph program (e.g. text-encoder +
denoiser + VAE-decoder for diffusion) gets one workspace per
compute graph without cross-graph aliasing.
*/
type WorkspaceMap struct {
	// portsByGraphNode resolves dispatch-time queries — given a graph
	// name and the dispatching ast.GraphNode, return the planner-
	// allocated tensor for its single output.
	outputs map[string]map[string]tensor.Tensor
	// inputs[graphName][nodeID] is the slice of input tensors in the
	// node's declared order. The dispatcher reads these to hydrate per-
	// node operand pointers.
	inputs map[string]map[string][]tensor.Tensor
	// boundaryInputs[graphName][inputName] resolves graph-level inputs
	// (declared in ast.Graph.Inputs) to their workspace tensor so the
	// host can write into the right slot before each forward pass.
	boundaryInputs map[string]map[string]tensor.Tensor

	workspaces map[string]*Workspace
}

/*
NewWorkspaceMap constructs an empty map. Callers populate it via
Attach once per (graph, planner topology) pair.
*/
func NewWorkspaceMap() *WorkspaceMap {
	return &WorkspaceMap{
		outputs:        make(map[string]map[string]tensor.Tensor),
		inputs:         make(map[string]map[string][]tensor.Tensor),
		boundaryInputs: make(map[string]map[string]tensor.Tensor),
		workspaces:     make(map[string]*Workspace),
	}
}

/*
Attach takes one compiled graph and the *ir.Topology the planner
produced for it, allocates a Workspace, and pre-resolves every
ast.GraphNode's input / output tensors into the per-graph maps the
dispatcher reads.

The bridge between ast.GraphNode and ir.Node is positional: the planner
topology was built by TopologyForPlanning, which walks ast.Graph.Nodes
in order and emits one ir.Node per ast.GraphNode at the same index.
This Attach honours that ordering when matching the two together.
*/
func (workspaceMap *WorkspaceMap) Attach(
	graphName string,
	graph *ast.Graph,
	topology *ir.Topology,
) error {
	if workspaceMap == nil {
		return fmt.Errorf("execution: nil workspace map")
	}

	if graph == nil || topology == nil {
		return fmt.Errorf("execution: attach graph %q: graph and topology required", graphName)
	}

	if len(graph.Nodes) != len(topology.Nodes) {
		return fmt.Errorf(
			"execution: attach graph %q: ast.Graph has %d nodes but planner topology has %d",
			graphName, len(graph.Nodes), len(topology.Nodes),
		)
	}

	workspace, err := NewWorkspace(topology)

	if err != nil {
		return fmt.Errorf("execution: attach graph %q: %w", graphName, err)
	}

	outputs := make(map[string]tensor.Tensor, len(graph.Nodes))
	inputs := make(map[string][]tensor.Tensor, len(graph.Nodes))
	boundary := make(map[string]tensor.Tensor)

	for nodeIndex, astNode := range graph.Nodes {
		irNode := topology.Nodes[nodeIndex]

		if len(irNode.Outputs) > 0 {
			outputTensor, err := workspace.TensorByPortID(irNode.Outputs[0].ID)

			if err != nil {
				_ = workspace.Close()

				return fmt.Errorf(
					"execution: attach graph %q node %q output: %w",
					graphName, astNode.ID, err,
				)
			}

			outputs[astNode.ID] = outputTensor
		}

		inputTensors := make([]tensor.Tensor, len(astNode.Inputs))

		for slotIndex, producerName := range astNode.Inputs {
			irPort := irNode.Inputs[slotIndex]

			if irPort == nil {
				_ = workspace.Close()

				return fmt.Errorf(
					"execution: attach graph %q node %q input slot %d: nil port",
					graphName, astNode.ID, slotIndex,
				)
			}

			inputTensor, err := workspace.TensorByPortID(irPort.ID)

			if err != nil {
				_ = workspace.Close()

				return fmt.Errorf(
					"execution: attach graph %q node %q input %q: %w",
					graphName, astNode.ID, producerName, err,
				)
			}

			inputTensors[slotIndex] = inputTensor

			// Boundary inputs (those whose name doesn't match any
			// producer) get an entry in boundary so the host can write
			// the per-call input before each forward pass.
			if isBoundaryInput(graph, producerName) {
				boundary[producerName] = inputTensor
			}
		}

		inputs[astNode.ID] = inputTensors
	}

	workspaceMap.outputs[graphName] = outputs
	workspaceMap.inputs[graphName] = inputs
	workspaceMap.boundaryInputs[graphName] = boundary
	workspaceMap.workspaces[graphName] = workspace

	return nil
}

/*
OutputFor returns the pre-resolved output tensor for one ast.GraphNode
in a compiled graph. The dispatcher calls this in place of
dispatcher.memory.Upload(...) for output allocation: the tensor is
already alive, sized, and aliased to its planned workspace slot.
*/
func (workspaceMap *WorkspaceMap) OutputFor(graphName, nodeID string) (tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	graphOutputs, ok := workspaceMap.outputs[graphName]

	if !ok {
		return nil, false
	}

	t, ok := graphOutputs[nodeID]

	return t, ok
}

/*
InputsFor returns the input tensors for one ast.GraphNode in the order
its Inputs slice declared. Used by handlers that need to know more
than just the value-table-resolved tensor for a producer (e.g., for
boundary-input nodes where there is no upstream producer in the
dispatcher's value table).
*/
func (workspaceMap *WorkspaceMap) InputsFor(graphName, nodeID string) ([]tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	graphInputs, ok := workspaceMap.inputs[graphName]

	if !ok {
		return nil, false
	}

	t, ok := graphInputs[nodeID]

	return t, ok
}

/*
BoundaryInput returns the workspace tensor for one graph-level input
declaration. The host writes into this tensor before each dispatch;
the dispatcher reads its value from the same storage.
*/
func (workspaceMap *WorkspaceMap) BoundaryInput(graphName, inputName string) (tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	boundary, ok := workspaceMap.boundaryInputs[graphName]

	if !ok {
		return nil, false
	}

	t, ok := boundary[inputName]

	return t, ok
}

/*
Close releases every attached workspace's storage back to the tensor
allocator. Idempotent; safe to call from defer chains during session
teardown.
*/
func (workspaceMap *WorkspaceMap) Close() error {
	if workspaceMap == nil {
		return nil
	}

	for _, workspace := range workspaceMap.workspaces {
		_ = workspace.Close()
	}

	workspaceMap.workspaces = nil
	workspaceMap.outputs = nil
	workspaceMap.inputs = nil
	workspaceMap.boundaryInputs = nil

	return nil
}

func isBoundaryInput(graph *ast.Graph, name string) bool {
	for _, declared := range graph.Inputs {
		if declared == name {
			return true
		}
	}

	return false
}
