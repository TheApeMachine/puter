//go:build xla

package xla

/*
NewBackend constructs an XLA backend. Returns ErrNeedsPlatformSetup
if the PJRT plugin cannot be opened on the current platform.
*/
func NewBackend() (*Backend, error) {
	backend := &Backend{
		workspace: NewWorkspace(),
		builder:   NewRuntimeBuilder(),
	}
	bridge, err := openXLABridge(backend)

	if err != nil {
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge, builder: backend.builder}
	backend.bindFamilies(computeHost)

	return backend, nil
}
