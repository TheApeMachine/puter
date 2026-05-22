//go:build darwin && cgo

package metal

import "fmt"

/*
LayerRunner batches one execution layer into a single Metal command
buffer and waits once at the boundary.
*/
type LayerRunner struct {
	backend *Backend
}

/*
NewLayerRunner returns a layer runner for the given Metal backend.
*/
func NewLayerRunner(backend *Backend) *LayerRunner {
	return &LayerRunner{backend: backend}
}

/*
Run executes run inside one Metal layer command buffer.
*/
func (layerRunner *LayerRunner) Run(run func() error) error {
	if layerRunner == nil || layerRunner.backend == nil {
		return run()
	}

	layerRunner.backend.BeginBatch()

	runErr := run()
	endErr := layerRunner.backend.EndBatch()

	if runErr != nil {
		return runErr
	}

	if endErr != nil {
		return fmt.Errorf("metal layer end: %w", endErr)
	}

	return nil
}
