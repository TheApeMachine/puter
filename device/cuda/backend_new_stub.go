//go:build !cuda

package cuda

/*
NewBackend constructs a CUDA backend. Returns ErrNeedsPlatformSetup
when built without the cuda tag.
*/
func NewBackend() (*Backend, error) {
	return nil, openCUDABridgeUnavailable()
}

func openCUDABridgeUnavailable() error {
	_, err := openCUDABridge(nil)
	return err
}
