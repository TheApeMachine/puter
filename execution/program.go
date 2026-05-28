package execution

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/runtime"
)

type compiledNodeRunner func(*dispatcher, *compiledNode) error

type compiledNode struct {
	node              *ast.GraphNode
	bind              OperationBind
	run               compiledNodeRunner
	inputSlots        []int
	outputSlot        int
	readsDeviceScalar bool
}

type executionProgram struct {
	layers     [][]*compiledNode
	slotByName map[string]int
	slotNames  []string
}

func compileExecutionProgram(
	graph *ast.Graph,
	plan *runtime.ExecutionPlan,
) (*executionProgram, error) {
	if graph == nil {
		return nil, fmt.Errorf("execution: graph is required")
	}

	if plan == nil {
		return nil, fmt.Errorf("execution: plan is required")
	}

	registry, err := defaultOperationRegistry()

	if err != nil {
		return nil, err
	}

	program := newExecutionProgram(graph)
	nodesByID := graphNodesByID(graph)

	for layerIndex, layer := range plan.Layers {
		compiledLayer, err := program.compileLayer(registry, nodesByID, layerIndex, layer)

		if err != nil {
			return nil, err
		}

		program.layers = append(program.layers, compiledLayer)
	}

	return program, nil
}

func newExecutionProgram(graph *ast.Graph) *executionProgram {
	program := &executionProgram{
		slotByName: make(map[string]int),
	}

	for _, input := range graph.Inputs {
		program.slotFor(input)
	}

	return program
}

func graphNodesByID(graph *ast.Graph) map[string]*ast.GraphNode {
	nodesByID := make(map[string]*ast.GraphNode, len(graph.Nodes))

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		nodesByID[node.ID] = node
	}

	return nodesByID
}

func (program *executionProgram) compileLayer(
	registry *operationRegistry,
	nodesByID map[string]*ast.GraphNode,
	layerIndex int,
	layer []string,
) ([]*compiledNode, error) {
	compiledLayer := make([]*compiledNode, 0, len(layer))

	for _, nodeID := range layer {
		node, ok := nodesByID[nodeID]

		if !ok {
			return nil, fmt.Errorf(
				"execution: plan layer %d references unknown node %q",
				layerIndex, nodeID,
			)
		}

		step, err := program.compileNode(registry, node)

		if err != nil {
			return nil, err
		}

		compiledLayer = append(compiledLayer, step)
	}

	return compiledLayer, nil
}

func (program *executionProgram) compileNode(
	registry *operationRegistry,
	node *ast.GraphNode,
) (*compiledNode, error) {
	step := &compiledNode{
		node:              node,
		inputSlots:        program.inputSlots(node),
		outputSlot:        program.slotFor(node.ID),
		readsDeviceScalar: nodeReadsDeviceScalar(node),
	}

	if node.Op == optimizer.FuseOp {
		step.run = runCompiledFusedNode
		return step, nil
	}

	if node.Op == "value.assign" {
		step.run = runCompiledAssignNode
		return step, nil
	}

	bind, err := registry.Bind(node)

	if err != nil {
		return nil, err
	}

	step.bind = bind
	step.run = runCompiledBoundNode

	return step, nil
}

func (program *executionProgram) inputSlots(node *ast.GraphNode) []int {
	slots := make([]int, len(node.Inputs))

	for inputIndex, inputName := range node.Inputs {
		slots[inputIndex] = program.slotFor(inputName)
	}

	return slots
}

func (program *executionProgram) slotFor(name string) int {
	if slot, ok := program.slotByName[name]; ok {
		return slot
	}

	slot := len(program.slotNames)
	program.slotByName[name] = slot
	program.slotNames = append(program.slotNames, name)

	return slot
}

func (program *executionProgram) run(dispatcher *dispatcher) error {
	for layerIndex, layer := range program.layers {
		if err := program.runLayer(dispatcher, layerIndex, layer); err != nil {
			return err
		}
	}

	return nil
}

func (program *executionProgram) runLayer(
	dispatcher *dispatcher,
	layerIndex int,
	layer []*compiledNode,
) error {
	if len(layer) == 1 {
		return program.runStep(dispatcher, layerIndex, layer[0])
	}

	return program.runConcurrentLayer(dispatcher, layerIndex, layer)
}

func (program *executionProgram) runConcurrentLayer(
	dispatcher *dispatcher,
	layerIndex int,
	layer []*compiledNode,
) error {
	var waitGroup sync.WaitGroup
	errs := make(chan error, len(layer))

	for _, step := range layer {
		waitGroup.Add(1)

		go func(step *compiledNode) {
			defer waitGroup.Done()

			if err := program.runStep(dispatcher, layerIndex, step); err != nil {
				errs <- err
			}
		}(step)
	}

	waitGroup.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			return err
		}
	}

	return nil
}

func (program *executionProgram) runStep(
	dispatcher *dispatcher,
	layerIndex int,
	step *compiledNode,
) error {
	if step == nil || step.run == nil {
		return fmt.Errorf("execution: plan layer %d contains an empty step", layerIndex)
	}

	if err := step.run(dispatcher, step); err != nil {
		return fmt.Errorf("execution: node %q (%s): %w", step.node.ID, step.node.Op, err)
	}

	return nil
}

func runCompiledBoundNode(dispatcher *dispatcher, step *compiledNode) error {
	return runBoundNodeWithSlots(
		dispatcher,
		step.node,
		step.bind,
		step.inputSlots,
		step.outputSlot,
	)
}

func runCompiledAssignNode(dispatcher *dispatcher, step *compiledNode) error {
	return dispatcher.runAssignWithSlots(step.node, step.inputSlots, step.outputSlot)
}

func runCompiledFusedNode(dispatcher *dispatcher, step *compiledNode) error {
	return dispatcher.runFusedNodeWithSlots(step.node, step.inputSlots, step.outputSlot)
}
