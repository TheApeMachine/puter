//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalActivationArena struct{}

func (bridge *metalBridge) beginGraphExecution() {}

func (bridge *metalBridge) endGraphExecution() {}

func (bridge *metalBridge) allocFromActivationArena(
	shape tensor.Shape,
	storageDType dtype.DType,
) (*metalTensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}
