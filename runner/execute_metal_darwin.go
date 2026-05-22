//go:build darwin && cgo

package runner

import (
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func graphActivationPlannerFor(
	location tensor.Location,
	memory tensor.Backend,
) graphActivationPlanner {
	if location != tensor.Metal {
		return nil
	}

	planner, ok := memory.(graphActivationPlanner)

	if !ok {
		return nil
	}

	return planner
}

/*
graphLayerRunner executes one plan layer inside a device command buffer.
*/
type graphLayerRunner interface {
	Run(func() error) error
}

func metalLayerRunnerFor(
	location tensor.Location,
	memory tensor.Backend,
) graphLayerRunner {
	if location != tensor.Metal {
		return nil
	}

	backend, ok := memory.(*metal.Backend)

	if !ok {
		return nil
	}

	return metal.NewLayerRunner(backend)
}
