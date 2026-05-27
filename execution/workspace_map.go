package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

/*
WorkspaceMap indexes per-graph workspaces by name. Backend stores one
per CallGraph entrypoint so a multi-graph program gets one workspace per
compute graph without cross-graph aliasing.
*/
type WorkspaceMap struct {
	outputs        map[string]map[string]tensor.Tensor
	inputs         map[string]map[string][]tensor.Tensor
	outputTypes    map[string]map[string]ir.PortType
	inputTypes     map[string]map[string][]ir.PortType
	boundaryInputs map[string]map[string]tensor.Tensor
	workspaces     map[string]*Workspace
}

/*
NewWorkspaceMap constructs an empty map. Callers populate it via Attach once
per graph and planner topology pair.
*/
func NewWorkspaceMap() *WorkspaceMap {
	return &WorkspaceMap{
		outputs:        make(map[string]map[string]tensor.Tensor),
		inputs:         make(map[string]map[string][]tensor.Tensor),
		outputTypes:    make(map[string]map[string]ir.PortType),
		inputTypes:     make(map[string]map[string][]ir.PortType),
		boundaryInputs: make(map[string]map[string]tensor.Tensor),
		workspaces:     make(map[string]*Workspace),
	}
}

/*
Attach takes one compiled graph and the topology the planner produced for it.
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

	return workspaceMap.attachWorkspace(graphName, graph, topology, workspace)
}

/*
MaxBindings returns the planner SymbolMap used to size one graph's workspace.
*/
func (workspaceMap *WorkspaceMap) MaxBindings(graphName string) ir.SymbolMap {
	if workspaceMap == nil {
		return nil
	}

	workspace, ok := workspaceMap.workspaces[graphName]

	if !ok || workspace == nil {
		return nil
	}

	return workspace.Layout().Bindings
}

/*
OutputFor returns the pre-resolved output tensor for one ast.GraphNode.
*/
func (workspaceMap *WorkspaceMap) OutputFor(graphName, nodeID string) (tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	graphOutputs, ok := workspaceMap.outputs[graphName]

	if !ok {
		return nil, false
	}

	value, ok := graphOutputs[nodeID]

	return value, ok
}

/*
InputsFor returns the input tensors for one ast.GraphNode in declared order.
*/
func (workspaceMap *WorkspaceMap) InputsFor(graphName, nodeID string) ([]tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	graphInputs, ok := workspaceMap.inputs[graphName]

	if !ok {
		return nil, false
	}

	value, ok := graphInputs[nodeID]

	return value, ok
}

/*
OutputTypeFor returns the planned PortType for one ast.GraphNode output.
*/
func (workspaceMap *WorkspaceMap) OutputTypeFor(graphName, nodeID string) (ir.PortType, bool) {
	if workspaceMap == nil {
		return ir.PortType{}, false
	}

	graphOutputTypes, ok := workspaceMap.outputTypes[graphName]

	if !ok {
		return ir.PortType{}, false
	}

	value, ok := graphOutputTypes[nodeID]

	return value, ok
}

/*
InputTypesFor returns the planned PortTypes for one ast.GraphNode's inputs.
*/
func (workspaceMap *WorkspaceMap) InputTypesFor(graphName, nodeID string) ([]ir.PortType, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	graphInputTypes, ok := workspaceMap.inputTypes[graphName]

	if !ok {
		return nil, false
	}

	value, ok := graphInputTypes[nodeID]

	return value, ok
}

/*
BoundaryInput returns the workspace tensor for one graph-level input.
*/
func (workspaceMap *WorkspaceMap) BoundaryInput(graphName, inputName string) (tensor.Tensor, bool) {
	if workspaceMap == nil {
		return nil, false
	}

	boundary, ok := workspaceMap.boundaryInputs[graphName]

	if !ok {
		return nil, false
	}

	value, ok := boundary[inputName]

	return value, ok
}

/*
Close releases every attached workspace's storage.
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
	workspaceMap.outputTypes = nil
	workspaceMap.inputTypes = nil
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
