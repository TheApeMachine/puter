package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

type workspaceSlotAllocator interface {
	AllocateWorkspaceSlot(byteCount int) (tensor.Tensor, error)
	ViewWorkspaceSlot(
		slot tensor.Tensor,
		shape tensor.Shape,
		elementFormat dtype.DType,
		byteCount int,
	) (tensor.Tensor, error)
}

func NewResidentWorkspace(topology *ir.Topology, memory tensor.Backend) (*Workspace, error) {
	if memory == nil {
		return nil, fmt.Errorf("execution: workspace memory backend is required")
	}

	if memory.Location() == tensor.Host {
		return NewWorkspace(topology)
	}

	allocator, ok := memory.(workspaceSlotAllocator)

	if !ok {
		return nil, fmt.Errorf(
			"execution: %s backend cannot allocate planned workspace slots",
			memory.Location(),
		)
	}

	return newWorkspaceFromSlots(topology, allocator)
}

func (workspaceMap *WorkspaceMap) AttachResident(
	graphName string,
	graph *ast.Graph,
	topology *ir.Topology,
	memory tensor.Backend,
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

	workspace, err := NewResidentWorkspace(topology, memory)

	if err != nil {
		return fmt.Errorf("execution: attach graph %q: %w", graphName, err)
	}

	return workspaceMap.attachWorkspace(graphName, graph, topology, workspace)
}

func (workspaceMap *WorkspaceMap) attachWorkspace(
	graphName string,
	graph *ast.Graph,
	topology *ir.Topology,
	workspace *Workspace,
) error {
	outputs := make(map[string]tensor.Tensor, len(graph.Nodes))
	inputs := make(map[string][]tensor.Tensor, len(graph.Nodes))
	outputTypes := make(map[string]ir.PortType, len(graph.Nodes))
	inputTypes := make(map[string][]ir.PortType, len(graph.Nodes))
	boundary := make(map[string]tensor.Tensor)

	for nodeIndex, astNode := range graph.Nodes {
		if err := workspaceMap.attachWorkspaceNode(
			graphName,
			graph,
			topology.Nodes[nodeIndex],
			astNode,
			workspace,
			outputs,
			inputs,
			outputTypes,
			inputTypes,
			boundary,
		); err != nil {
			_ = workspace.Close()

			return err
		}
	}

	workspaceMap.ensureTypeMaps()
	workspaceMap.outputs[graphName] = outputs
	workspaceMap.inputs[graphName] = inputs
	workspaceMap.outputTypes[graphName] = outputTypes
	workspaceMap.inputTypes[graphName] = inputTypes
	workspaceMap.boundaryInputs[graphName] = boundary
	workspaceMap.workspaces[graphName] = workspace

	return nil
}

func (workspaceMap *WorkspaceMap) ensureTypeMaps() {
	if workspaceMap.outputTypes == nil {
		workspaceMap.outputTypes = make(map[string]map[string]ir.PortType)
	}

	if workspaceMap.inputTypes == nil {
		workspaceMap.inputTypes = make(map[string]map[string][]ir.PortType)
	}
}

func (workspaceMap *WorkspaceMap) attachWorkspaceNode(
	graphName string,
	graph *ast.Graph,
	irNode *ir.Node,
	astNode *ast.GraphNode,
	workspace *Workspace,
	outputs map[string]tensor.Tensor,
	inputs map[string][]tensor.Tensor,
	outputTypes map[string]ir.PortType,
	inputTypes map[string][]ir.PortType,
	boundary map[string]tensor.Tensor,
) error {
	if len(irNode.Outputs) > 0 {
		outputTensor, err := workspace.TensorByPortID(irNode.Outputs[0].ID)

		if err != nil {
			return fmt.Errorf(
				"execution: attach graph %q node %q output: %w",
				graphName, astNode.ID, err,
			)
		}

		outputs[astNode.ID] = outputTensor
		outputTypes[astNode.ID] = irNode.Outputs[0].Type
	}

	inputTensors := make([]tensor.Tensor, len(astNode.Inputs))
	inputPortTypes := make([]ir.PortType, len(astNode.Inputs))

	for slotIndex, producerName := range astNode.Inputs {
		inputTensor, err := workspaceMap.attachWorkspaceInput(
			graphName,
			graph,
			irNode,
			astNode,
			workspace,
			slotIndex,
			producerName,
			boundary,
		)

		if err != nil {
			return err
		}

		inputTensors[slotIndex] = inputTensor
		inputPortTypes[slotIndex] = irNode.Inputs[slotIndex].Type
	}

	inputs[astNode.ID] = inputTensors
	inputTypes[astNode.ID] = inputPortTypes

	return nil
}

func (workspaceMap *WorkspaceMap) attachWorkspaceInput(
	graphName string,
	graph *ast.Graph,
	irNode *ir.Node,
	astNode *ast.GraphNode,
	workspace *Workspace,
	slotIndex int,
	producerName string,
	boundary map[string]tensor.Tensor,
) (tensor.Tensor, error) {
	irPort := irNode.Inputs[slotIndex]

	if irPort == nil {
		return nil, fmt.Errorf(
			"execution: attach graph %q node %q input slot %d: nil port",
			graphName, astNode.ID, slotIndex,
		)
	}

	inputTensor, err := workspace.TensorByPortID(irPort.ID)

	if err != nil {
		return nil, fmt.Errorf(
			"execution: attach graph %q node %q input %q: %w",
			graphName, astNode.ID, producerName, err,
		)
	}

	if isBoundaryInput(graph, producerName) {
		boundary[producerName] = inputTensor
	}

	return inputTensor, nil
}

func newWorkspaceFromSlots(
	topology *ir.Topology,
	allocator workspaceSlotAllocator,
) (*Workspace, error) {
	if topology == nil {
		return nil, fmt.Errorf("execution: workspace topology is required")
	}

	layout := topology.Workspace

	if layout.Size <= 0 {
		return nil, fmt.Errorf(
			"execution: workspace layout has zero size — was PlanWorkspace run?",
		)
	}

	slots, err := allocateWorkspaceSlots(layout, allocator)

	if err != nil {
		return nil, err
	}

	workspace := &Workspace{
		layout:  layout,
		tensors: make(map[int32]tensor.Tensor, len(layout.Allocations)),
		slots:   slots,
	}

	if err := workspace.preBuildResidentTensors(topology, allocator); err != nil {
		_ = workspace.Close()

		return nil, err
	}

	return workspace, nil
}

func allocateWorkspaceSlots(
	layout ir.WorkspaceLayout,
	allocator workspaceSlotAllocator,
) (map[int64]tensor.Tensor, error) {
	slotSizes := make(map[int64]int64)

	for _, interval := range layout.Allocations {
		if interval.Size <= slotSizes[interval.Offset] {
			continue
		}

		slotSizes[interval.Offset] = interval.Size
	}

	slots := make(map[int64]tensor.Tensor, len(slotSizes))

	for offset, size := range slotSizes {
		slot, err := allocator.AllocateWorkspaceSlot(int(size))

		if err != nil {
			closeWorkspaceSlots(slots)

			return nil, fmt.Errorf("execution: allocate workspace slot %d: %w", offset, err)
		}

		slots[offset] = slot
	}

	return slots, nil
}

func (workspace *Workspace) preBuildResidentTensors(
	topology *ir.Topology,
	allocator workspaceSlotAllocator,
) error {
	allocationByPortID := make(map[int32]ir.Interval, len(topology.Workspace.Allocations))

	for _, interval := range topology.Workspace.Allocations {
		allocationByPortID[interval.PortID] = interval
	}

	return workspace.materializeResidentTopology(topology, allocationByPortID, workspace.slots, allocator)
}

func (workspace *Workspace) materializeResidentTopology(
	topology *ir.Topology,
	allocationByPortID map[int32]ir.Interval,
	slots map[int64]tensor.Tensor,
	allocator workspaceSlotAllocator,
) error {
	for _, node := range topology.Nodes {
		for _, port := range node.Outputs {
			if err := workspace.materializeResidentPort(port, allocationByPortID, slots, allocator); err != nil {
				return err
			}
		}

		for _, port := range node.Inputs {
			if err := workspace.materializeResidentPort(port, allocationByPortID, slots, allocator); err != nil {
				return err
			}
		}
	}

	return nil
}

func (workspace *Workspace) materializeResidentPort(
	port *ir.Port,
	allocationByPortID map[int32]ir.Interval,
	slots map[int64]tensor.Tensor,
	allocator workspaceSlotAllocator,
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

	slot, ok := slots[interval.Offset]

	if !ok {
		return fmt.Errorf("execution: port %d workspace slot %d missing", port.ID, interval.Offset)
	}

	shape, err := resolveShape(port.Type.ShapeSchema, port.Type.DType, workspace.layout.Bindings)

	if err != nil {
		return fmt.Errorf("execution: port %d shape: %w", port.ID, err)
	}

	byteCount, err := shape.Bytes(port.Type.DType)

	if err != nil {
		return fmt.Errorf("execution: port %d byte size: %w", port.ID, err)
	}

	workspace.tensors[port.ID], err = allocator.ViewWorkspaceSlot(
		slot,
		shape,
		port.Type.DType,
		byteCount,
	)

	if err != nil {
		return fmt.Errorf("execution: port %d workspace view: %w", port.ID, err)
	}

	return nil
}
