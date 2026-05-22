//go:build !darwin || !cgo

package metal

/*
LayerRunner is unavailable without darwin+cgo.
*/
type LayerRunner struct {
	backend *Backend
}

func NewLayerRunner(backend *Backend) *LayerRunner {
	return &LayerRunner{backend: backend}
}

func (layerRunner *LayerRunner) Run(run func() error) error {
	return run()
}
