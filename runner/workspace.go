package runner

import (
	"sync"

	"github.com/theapemachine/manifesto/tensor"
)

/*
workspace stores resident tensors keyed by graph node ID.
*/
type workspace struct {
	mu      sync.Mutex
	tensors map[string]tensor.Tensor
}

func newWorkspace() *workspace {
	return &workspace{
		tensors: make(map[string]tensor.Tensor),
	}
}

func (tensorWorkspace *workspace) Store(nodeID string, value tensor.Tensor) {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	if existing, ok := tensorWorkspace.tensors[nodeID]; ok && existing != value {
		_ = existing.Close()
	}

	tensorWorkspace.tensors[nodeID] = value
}

func (tensorWorkspace *workspace) Load(nodeID string) (tensor.Tensor, bool) {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	value, ok := tensorWorkspace.tensors[nodeID]

	return value, ok
}

func (tensorWorkspace *workspace) Close() {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	for nodeID, value := range tensorWorkspace.tensors {
		_ = value.Close()
		delete(tensorWorkspace.tensors, nodeID)
	}
}
