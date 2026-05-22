package runner

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

/*
graphCommandBatcher batches device dispatches into one command buffer per layer.
Metal implements this; host backends execute nodes without batching.
*/
type graphCommandBatcher interface {
	BeginBatch()
	EndBatch() error
}

/*
graphActivationPlanner resets device-side activation arenas per forward pass.
*/
type graphActivationPlanner interface {
	BeginGraphExecution()
	EndGraphExecution()
}

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
	if plan == nil {
		var err error

		plan, err = runtime.NewExecutionPlan("graph", computeGraph)

		if err != nil {
			return err
		}
	}

	remainingUses := tensorUseCounts(computeGraph)
	batcher := graphCommandBatcherFor(location, memory)
	activationPlanner := graphActivationPlannerFor(location, memory)

	if activationPlanner != nil {
		activationPlanner.BeginGraphExecution()
		defer activationPlanner.EndGraphExecution()
	}

	layerRunner := metalLayerRunnerFor(location, memory)

	for _, layer := range plan.Layers {
		if err := ctx.Err(); err != nil {
			return err
		}

		if layerRunner != nil {
			if err := layerRunner.Run(func() error {
				return executePlanLayer(
					ctx,
					layer,
					computeGraph,
					location,
					memory,
					checkpointPath,
					weights,
					bindings,
					tensorWorkspace,
					workerPool,
				)
			}); err != nil {
				return err
			}
		} else {
			if batcher != nil {
				batcher.BeginBatch()
			}

			if err := executePlanLayer(
				ctx,
				layer,
				computeGraph,
				location,
				memory,
				checkpointPath,
				weights,
				bindings,
				tensorWorkspace,
				workerPool,
			); err != nil {
				if batcher != nil {
					_ = batcher.EndBatch()
				}

				return err
			}

			if batcher != nil {
				if err := batcher.EndBatch(); err != nil {
					return err
				}
			}
		}

		for _, nodeID := range layer {
			node := findComputeNode(computeGraph, nodeID)

			if node == nil {
				return fmt.Errorf("runner: missing node %q", nodeID)
			}

			releaseConsumedTensors(node, remainingUses, tensorWorkspace)
		}
	}

	return nil
}

func graphCommandBatcherFor(
	location tensor.Location,
	memory tensor.Backend,
) graphCommandBatcher {
	if location != tensor.Metal {
		return nil
	}

	batcher, ok := memory.(graphCommandBatcher)

	if !ok {
		return nil
	}

	return batcher
}

func executePlanLayer(
	ctx context.Context,
	layer []string,
	computeGraph *ir.Graph,
	location tensor.Location,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
	workerPool *qpool.Q,
) error {
	if len(layer) == 0 {
		return nil
	}

	if workerPool != nil && len(layer) > 1 && location != tensor.Metal {
		return executePlanLayerConcurrent(
			ctx,
			layer,
			computeGraph,
			location,
			memory,
			checkpointPath,
			weights,
			bindings,
			tensorWorkspace,
			workerPool,
		)
	}

	return executePlanLayerSequential(
		layer,
		computeGraph,
		location,
		memory,
		checkpointPath,
		weights,
		bindings,
		tensorWorkspace,
	)
}

func executePlanLayerSequential(
	layer []string,
	computeGraph *ir.Graph,
	location tensor.Location,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
) error {
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

	return nil
}

func executePlanLayerConcurrent(
	ctx context.Context,
	layer []string,
	computeGraph *ir.Graph,
	location tensor.Location,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
	workerPool *qpool.Q,
) error {
	pending := make([]chan *qpool.QValue[any], 0, len(layer))

	for _, nodeID := range layer {
		node := findComputeNode(computeGraph, nodeID)

		if node == nil {
			return fmt.Errorf("runner: missing node %q", nodeID)
		}

		pending = append(pending, workerPool.ScheduleFast(ctx, func(jobCtx context.Context) (any, error) {
			_ = jobCtx

			dispatchErr := dispatchNode(
				node,
				location,
				memory,
				checkpointPath,
				weights,
				bindings,
				tensorWorkspace,
			)

			return nil, dispatchErr
		}))
	}

	for index, result := range pending {
		select {
		case qvalue := <-result:
			if qvalue.Error != nil {
				return fmt.Errorf("runner: node %q: %w", layer[index], qvalue.Error)
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

func tensorUseCounts(computeGraph *ir.Graph) map[string]int {
	counts := make(map[string]int)

	for _, node := range computeGraph.Nodes() {
		for _, inputNode := range node.Inputs() {
			counts[inputNode.ID()]++
		}
	}

	return counts
}

func releaseConsumedTensors(
	node *ir.Node,
	remainingUses map[string]int,
	tensorWorkspace *workspace,
) {
	for _, inputNode := range node.Inputs() {
		inputID := inputNode.ID()
		remainingUses[inputID]--

		if remainingUses[inputID] > 0 {
			continue
		}

		tensorWorkspace.ReleaseOwned(inputID)
	}
}

func findComputeNode(computeGraph *ir.Graph, nodeID string) *ir.Node {
	for _, node := range computeGraph.Nodes() {
		if node.ID() == nodeID {
			return node
		}
	}

	return nil
}
