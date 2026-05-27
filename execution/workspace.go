package execution

import (
	"fmt"

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
	slots   map[int64]tensor.Tensor
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

	shape, err := resolveShape(port.Type.ShapeSchema, port.Type.DType, workspace.layout.Bindings)

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
	if workspace == nil {
		return nil
	}

	if workspace.storage != nil {
		tensor.Release(workspace.storage)
		workspace.storage = nil
	}

	closeWorkspaceSlots(workspace.slots)
	workspace.slots = nil
	workspace.tensors = nil

	return nil
}

func closeWorkspaceSlots(slots map[int64]tensor.Tensor) {
	for _, slot := range slots {
		_ = slot.Close()
	}
}
