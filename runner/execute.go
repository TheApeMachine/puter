package runner

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

func executeGraph(
	ctx context.Context,
	computeGraph *ir.Graph,
	plan *runtime.ExecutionPlan,
	location tensor.Location,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
	workerPool *qpool.Q,
) error {
	_ = ctx
	_ = workerPool

	if plan == nil {
		var err error

		plan, err = runtime.NewExecutionPlan("graph", computeGraph)

		if err != nil {
			return err
		}
	}

	for _, layer := range plan.Layers {
		for _, nodeID := range layer {
			node := findComputeNode(computeGraph, nodeID)

			if node == nil {
				return fmt.Errorf("runner: missing node %q", nodeID)
			}

			if err := dispatchNode(
				node,
				location,
				memory,
				checkpointPath,
				weights,
				bindings,
				tensorWorkspace,
			); err != nil {
				return fmt.Errorf("runner: node %q: %w", node.ID(), err)
			}
		}
	}

	return nil
}

func findComputeNode(computeGraph *ir.Graph, nodeID string) *ir.Node {
	for _, node := range computeGraph.Nodes() {
		if node.ID() == nodeID {
			return node
		}
	}

	return nil
}
