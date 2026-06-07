//go:build darwin && cgo

package metal

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
WriteTensorBytes copies host bytes into one resident Metal tensor buffer.
*/
func (backend *Backend) WriteTensorBytes(target tensor.Tensor, bytesIn []byte) error {
	if backend.closed.Load() {
		return tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return tensor.ErrNeedsPlatformSetup
	}

	deviceTensor, err := requireDeviceTensor(target)

	if err != nil {
		return err
	}

	return backend.bridge.writeDeviceTensorBytes(deviceTensor, bytesIn)
}
