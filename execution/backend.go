package execution

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/puter/pool"
)

/*
Backend executes manifest graph.call steps through discovered device backends.
It implements manifesto/runtime.Backend and dispatches via device.Backend per ARCHITECTURE.md.
*/
type Backend struct {
	devicePool *pool.Pool
}

/*
New constructs an execution backend over a discovered device pool.
*/
func New(devicePool *pool.Pool) *Backend {
	return &Backend{devicePool: devicePool}
}

/*
Close releases backend-owned resources.
*/
func (backend *Backend) Close() error {
	return nil
}

/*
CallGraph executes one graph.call program step on the active device backend.
*/
func (backend *Backend) CallGraph(
	ctx context.Context,
	request runtime.GraphCallRequest,
) (runtime.GraphCallResult, error) {
	_ = ctx

	if backend == nil || backend.devicePool == nil {
		return runtime.GraphCallResult{}, fmt.Errorf("execution: device pool is required")
	}

	return runtime.GraphCallResult{}, fmt.Errorf(
		"execution: graph %q dispatch is not implemented",
		request.GraphName,
	)
}
