//go:build !darwin || !cgo

package runner

import (
	"github.com/theapemachine/manifesto/tensor"
)

func graphActivationPlannerFor(
	location tensor.Location,
	memory tensor.Backend,
) graphActivationPlanner {
	return nil
}

func metalLayerRunnerFor(
	location tensor.Location,
	memory tensor.Backend,
) graphLayerRunner {
	return nil
}
