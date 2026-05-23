//go:build cuda

package cuda

/*
NewBackend constructs a CUDA backend. Returns ErrNeedsPlatformSetup
if no CUDA-capable device is present or the cgo toolchain is missing.
*/
func NewBackend() (*Backend, error) {
	backend := &Backend{}
	bridge, err := openCUDABridge(backend)

	if err != nil {
		return nil, err
	}

	backend.bridge = bridge
	computeHost := &ComputeHost{bridge: bridge}
	backend.bindFamilies(computeHost)

	return backend, nil
}
