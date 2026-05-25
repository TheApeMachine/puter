package execution

import (
	"errors"

	"github.com/theapemachine/manifesto/tensor"
)

/*
ErrWeightNotFound is returned by WeightStore.Lookup when a requested tensor
name is not registered.
*/
var ErrWeightNotFound = errors.New("execution: weight not found")

/*
WeightStore resolves parameter names (as declared in ast.GraphNode.Weights)
into resident tensors. Implementations bridge to safetensors archives,
checkpoint shards, or memory-mapped weight pools.

The execution backend never loads weights itself — it asks WeightStore on
demand for each node and caches the returned handle for the duration of
the CallGraph invocation.
*/
type WeightStore interface {
	/*
		Lookup returns the resident tensor backing the given parameter
		name (e.g. "model.layers.0.input_layernorm.weight"). Implementations
		MUST return ErrWeightNotFound when the name has no associated
		tensor; the dispatcher uses this to distinguish missing weights
		from infrastructure errors.
	*/
	Lookup(name string) (tensor.Tensor, error)
}

/*
TransposedLookup is an optional capability advertised by a WeightStore
implementation. Callers that need the row-major transpose of a 2-D
weight (most commonly, projection.linear consumers that have to compute
y = x @ W.T for a HuggingFace-style nn.Linear weight stored as
[out_features, in_features]) type-assert to this interface, falling back
to a plain Lookup + an inline transpose when the implementation does not
provide one.

Implementations are encouraged to cache the materialized transpose so
the per-token forward pass does not pay the copy cost on every call.
*/
type TransposedLookup interface {
	Lookup(name string) (tensor.Tensor, error)
	LookupTransposed(name string) (tensor.Tensor, error)
}

/*
nilWeightStore is the default fallback when no weight store is injected.
It returns ErrWeightNotFound for every lookup so graphs with weighted
nodes fail loudly with a clear message rather than silently producing
zero output.
*/
type nilWeightStore struct{}

func (store nilWeightStore) Lookup(name string) (tensor.Tensor, error) {
	_ = name
	return nil, ErrWeightNotFound
}
