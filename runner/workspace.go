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
	owned   map[string]bool
}

func newWorkspace() *workspace {
	return &workspace{
		tensors: make(map[string]tensor.Tensor),
		owned:   make(map[string]bool),
	}
}

func (tensorWorkspace *workspace) Store(nodeID string, value tensor.Tensor) {
	tensorWorkspace.store(nodeID, value, true)
}

func (tensorWorkspace *workspace) StoreBorrowed(nodeID string, value tensor.Tensor) {
	tensorWorkspace.store(nodeID, value, false)
}

func (tensorWorkspace *workspace) store(nodeID string, value tensor.Tensor, owned bool) {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	if existing, ok := tensorWorkspace.tensors[nodeID]; ok && existing != value && tensorWorkspace.owned[nodeID] {
		_ = existing.Close()
	}

	tensorWorkspace.tensors[nodeID] = value
	tensorWorkspace.owned[nodeID] = owned
}

func (tensorWorkspace *workspace) Load(nodeID string) (tensor.Tensor, bool) {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	value, ok := tensorWorkspace.tensors[nodeID]

	return value, ok
}

func (tensorWorkspace *workspace) Detach(nodeID string) (tensor.Tensor, bool) {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	value, ok := tensorWorkspace.tensors[nodeID]

	if !ok {
		return nil, false
	}

	delete(tensorWorkspace.tensors, nodeID)
	delete(tensorWorkspace.owned, nodeID)

	return value, true
}

func (tensorWorkspace *workspace) Close() {
	tensorWorkspace.mu.Lock()
	defer tensorWorkspace.mu.Unlock()

	for nodeID, value := range tensorWorkspace.tensors {
		if tensorWorkspace.owned[nodeID] {
			_ = value.Close()
		}

		delete(tensorWorkspace.tensors, nodeID)
		delete(tensorWorkspace.owned, nodeID)
	}
}
