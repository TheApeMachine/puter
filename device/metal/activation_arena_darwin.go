//go:build darwin && cgo

package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalActivationArena struct {
	epoch uint64
}

func (bridge *metalBridge) beginGraphExecution() {
	bridge.arena.epoch++
}

func (bridge *metalBridge) endGraphExecution() {
	bridge.arena.epoch++
}

func (bridge *metalBridge) allocFromActivationArena(
	shape tensor.Shape,
	storageDType dtype.DType,
) (*metalTensor, error) {
	return bridge.empty(shape, storageDType)
}
