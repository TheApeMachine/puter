//go:build !xla

package xla

/*
NewBackend constructs an XLA backend. Returns ErrNeedsPlatformSetup
when built without the xla tag.
*/
func NewBackend() (*Backend, error) {
	return nil, openXLABridgeUnavailable()
}

func openXLABridgeUnavailable() error {
	_, err := openXLABridge(nil)
	return err
}
