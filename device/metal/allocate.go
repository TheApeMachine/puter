package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
NewZeroed allocates a Metal-resident tensor with zeroed storage.
*/
func (backend *Backend) NewZeroed(shape tensor.Shape, asType dtype.DType) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	target, err := backend.bridge.empty(shape, asType)

	if err != nil {
		return nil, err
	}

	if target.Bytes() == 0 {
		return target, nil
	}

	contents := metalTensorContents(target)

	if contents == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	metalMemset(contents, 0, target.Bytes())

	return target, nil
}
