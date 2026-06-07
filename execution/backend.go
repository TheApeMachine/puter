package execution

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/puter/pool"
)

/*
Backend executes manifest graph.call steps through discovered device backends.
It implements manifesto/runtime.Backend and dispatches via device.Backend per
ARCHITECTURE.md.
*/
type Backend struct {
	devicePool *pool.Pool
	weights    WeightStore
	workspaces *WorkspaceMap
}

/*
New constructs an execution backend over a discovered device pool.
*/
func New(devicePool *pool.Pool) *Backend {
	return &Backend{
		devicePool: devicePool,
		weights:    nilWeightStore{},
		workspaces: NewWorkspaceMap(),
	}
}

/*
WithWeights injects the weight store used to resolve safetensors-backed
parameters during graph.call dispatch. Returns the receiver so the call
can be chained from caramba's wire-up.
*/
func (backend *Backend) WithWeights(store WeightStore) *Backend {
	if backend == nil {
		return nil
	}

	if store == nil {
		store = nilWeightStore{}
	}

	backend.weights = store

	return backend
}

/*
Workspaces returns the backend's per-graph WorkspaceMap so the
orchestrator and session can register planner output before the first
dispatch. The map starts empty; callers attach one entry per compiled
graph via WorkspaceMap.Attach.
*/
func (backend *Backend) Workspaces() *WorkspaceMap {
	if backend == nil {
		return nil
	}

	return backend.workspaces
}

/*
AttachWorkspace satisfies manifesto/runtime.WorkspaceAttacher. The
orchestrator calls this once per compiled graph after the planner
populates the topology's WorkspaceLayout. The backend allocates resident
workspace storage on the pinned memory backend, pre-builds tensor handles
for every planned port, and indexes those tensors by ast.GraphNode ID so
the dispatcher can look up node outputs without per-call allocation.
*/
func (backend *Backend) AttachWorkspace(
	graphName string,
	graph *ast.Graph,
	topology *ir.Topology,
) error {
	if backend == nil {
		return fmt.Errorf("execution: nil backend")
	}

	if backend.workspaces == nil {
		backend.workspaces = NewWorkspaceMap()
	}

	memory, _, err := backend.devicePool.MemoryBackend()

	if err != nil {
		return err
	}

	return backend.workspaces.AttachResident(graphName, graph, topology, memory)
}

/*
Close releases backend-owned resources, including every attached
workspace's mmap/slab allocation. Safe to call multiple times; the
WorkspaceMap's own Close is idempotent.
*/
func (backend *Backend) Close() error {
	if backend == nil {
		return nil
	}

	if backend.workspaces != nil {
		_ = backend.workspaces.Close()
		backend.workspaces = nil
	}

	return nil
}

/*
CallGraph executes one graph.call program step on the active device backend.

The dispatcher walks request.Plan.Layers, looking up each node by ID in
request.Graph and routing to one of three execution paths:

 1. Fused nodes (Op == optimizer.FuseOp) run the codegen-attached
    CPUKernel directly.
 2. Known device ops (Embedding, RMSNorm, Matmul, Add, etc.) route to
    the active device.Backend method.
 3. Anything else returns a clear "unsupported op" error so missing
    coverage is visible.

Inputs declared on the graph.call step (request.Inputs) seed the value
table; the table grows as each node writes its output. At the end of the
walk, the named outputs (request.Graph.Outputs or, when absent, the
graph's logical sink) are collected into GraphCallResult.Outputs.
*/
func (backend *Backend) CallGraph(
	ctx context.Context,
	request runtime.GraphCallRequest,
) (runtime.GraphCallResult, error) {
	_ = ctx

	if backend == nil || backend.devicePool == nil {
		return runtime.GraphCallResult{}, fmt.Errorf("execution: device pool is required")
	}

	if request.Graph == nil {
		return runtime.GraphCallResult{}, fmt.Errorf(
			"execution: graph %q is missing the ast.Graph payload", request.GraphName,
		)
	}

	if request.Plan == nil {
		return runtime.GraphCallResult{}, fmt.Errorf(
			"execution: graph %q is missing an execution plan", request.GraphName,
		)
	}

	deviceBackend, err := backend.pickDevice()

	if err != nil {
		return runtime.GraphCallResult{}, err
	}

	memory, _, err := backend.devicePool.MemoryBackend()

	if err != nil {
		return runtime.GraphCallResult{}, err
	}

	weights := backend.weights

	if weights == nil {
		weights = nilWeightStore{}
	}

	dispatcher := newDispatcher(
		request.GraphName,
		request.Graph,
		request.Plan,
		deviceBackend,
		memory,
		weights,
		backend.workspaces,
		request.LaunchBindings,
	)

	seededInputs, err := seedGraphInputs(backend, request.GraphName, request.Inputs, memory)

	if err != nil {
		return runtime.GraphCallResult{}, fmt.Errorf(
			"execution: graph %q inputs: %w", request.GraphName, err,
		)
	}

	for name, value := range seededInputs {
		dispatcher.values.set(name, value)
	}

	if err := dispatcher.run(); err != nil {
		return runtime.GraphCallResult{}, fmt.Errorf(
			"execution: graph %q: %w", request.GraphName, err,
		)
	}

	outputs := backend.collectOutputs(dispatcher, request)

	return runtime.GraphCallResult{Outputs: outputs}, nil
}

/*
pickDevice returns the highest-precedence device backend on the pool. CPU
is always the safe fallback (the host backend implements device.Backend
fully); Metal is preferred when available on Darwin.
*/
func (backend *Backend) pickDevice() (executionDevice, error) {
	for _, deviceID := range backend.devicePool.DeviceIDs() {
		deviceBackend, err := backend.devicePool.Device(deviceID)

		if err != nil {
			continue
		}

		executionBackend, ok := deviceBackend.(executionDevice)

		if !ok {
			continue
		}

		return executionBackend, nil
	}

	return nil, fmt.Errorf("execution: no device backends registered")
}

/*
collectOutputs reads each declared output port out of the value table and
returns them under their public name. When the graph declares no outputs
the dispatcher emits the value table's final entry (typically the last
node's output).
*/
func (backend *Backend) collectOutputs(
	dispatcher *dispatcher,
	request runtime.GraphCallRequest,
) map[string]any {
	outputs := make(map[string]any, len(request.Graph.Outputs))

	if len(request.Graph.Outputs) > 0 {
		for name, ref := range request.Graph.Outputs {
			if value, ok := dispatcher.values.get(ref); ok {
				outputs[name] = value
			}
		}

		return outputs
	}

	// No explicit outputs: return the last node's value under its node ID.
	if len(request.Graph.Nodes) == 0 {
		return outputs
	}

	finalNode := request.Graph.Nodes[len(request.Graph.Nodes)-1]

	if value, ok := dispatcher.values.get(finalNode.ID); ok {
		outputs[finalNode.ID] = value
	}

	return outputs
}
