package runner

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/puter/pool"
)

/*
Runner executes graph.call steps via discovered device backends.
It implements manifesto/runtime.Backend.
*/
type Runner struct {
	devicePool *pool.Pool
}

/*
New constructs a graph runner over a discovered device pool.
*/
func New(devicePool *pool.Pool) *Runner {
	return &Runner{devicePool: devicePool}
}

/*
CallGraph executes one graph.call program step on the preferred compute device.
Independent nodes within each plan layer run concurrently through qpool when available.
*/
func (graphRunner *Runner) CallGraph(
	ctx context.Context,
	request runtime.GraphCallRequest,
) (runtime.GraphCallResult, error) {
	if request.GraphName == "" {
		return runtime.GraphCallResult{}, fmt.Errorf("runner: graph name is required")
	}

	if request.Graph == nil {
		return runtime.GraphCallResult{}, fmt.Errorf("runner: graph %q manifest graph is required", request.GraphName)
	}

	computeGraph, ok := request.Compute.(*ir.Graph)

	if !ok || computeGraph == nil {
		return runtime.GraphCallResult{}, fmt.Errorf("runner: graph %q compute graph is required", request.GraphName)
	}

	if graphRunner.devicePool == nil {
		return runtime.GraphCallResult{}, fmt.Errorf("runner: device pool is required")
	}

	memory, deviceID, err := graphRunner.devicePool.ComputeMemory()

	if err != nil {
		return runtime.GraphCallResult{}, fmt.Errorf("runner: resolve compute memory: %w", err)
	}

	tensorWorkspace := newWorkspace()
	defer tensorWorkspace.Close()

	weightTable := newWeightCache(memory)
	defer weightTable.Close()

	bindings := newManifestBindings(request.Graph)

	if err := bindProgramInputs(memory, request.Graph, computeGraph, request.Inputs, tensorWorkspace); err != nil {
		return runtime.GraphCallResult{}, err
	}

	checkpointPath := weightsPath(request.Graph)

	if err := executeGraph(
		ctx,
		computeGraph,
		request.Plan,
		deviceID.Location,
		memory,
		checkpointPath,
		weightTable,
		bindings,
		tensorWorkspace,
		graphRunner.devicePool.WorkerPool(),
	); err != nil {
		return runtime.GraphCallResult{}, err
	}

	outputs, err := collectProgramOutputs(memory, request.Graph, tensorWorkspace)

	if err != nil {
		return runtime.GraphCallResult{}, err
	}

	return runtime.GraphCallResult{Outputs: outputs}, nil
}
