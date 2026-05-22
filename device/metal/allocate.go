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

/*
NewEmpty allocates uninitialized Metal-resident storage. During graph
execution this draws from the activation bump arena; otherwise it uses
the buffer pool without zeroing host memory.
*/
func (backend *Backend) NewEmpty(shape tensor.Shape, asType dtype.DType) (tensor.Tensor, error) {
	if backend.closed.Load() {
		return nil, tensor.ErrBackendClosed
	}

	if backend.bridge == nil {
		return nil, tensor.ErrNeedsPlatformSetup
	}

	if backend.bridge.graphDepth > 0 {
		return backend.bridge.allocFromActivationArena(shape, asType)
	}

	return backend.bridge.empty(shape, asType)
}

/*
BeginGraphExecution resets the activation arena for one forward pass.
*/
func (backend *Backend) BeginGraphExecution() {
	if backend.bridge == nil {
		return
	}

	backend.bridge.graphDepth++

	if backend.bridge.graphDepth == 1 {
		backend.bridge.beginGraphExecution()
	}
}

/*
EndGraphExecution invalidates arena scratch tensors for the pass.
*/
func (backend *Backend) EndGraphExecution() {
	if backend.bridge == nil || backend.bridge.graphDepth == 0 {
		return
	}

	backend.bridge.graphDepth--

	if backend.bridge.graphDepth == 0 {
		backend.bridge.endGraphExecution()
	}
}
