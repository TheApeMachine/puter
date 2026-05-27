package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

/*
layerStorageView returns the per-layer slice of a paged KV tensor. Storage
may be rank 4 [pages, page_size, heads, dim] for single-layer graphs or
rank 5 [layers, pages, page_size, heads, dim] for stacked decoder caches.
*/
func layerStorageView(storage tensor.Tensor, layerIndex int) (tensor.Tensor, error) {
	if storage == nil {
		return nil, fmt.Errorf("layer storage view: storage is required")
	}

	dimensions := storage.Shape().Dims()

	if len(dimensions) == 4 {
		return storage, nil
	}

	if len(dimensions) != 5 {
		return nil, fmt.Errorf(
			"layer storage view: expected rank 4 or 5, got %d",
			len(dimensions),
		)
	}

	if layerIndex < 0 || layerIndex >= dimensions[0] {
		return nil, fmt.Errorf(
			"layer storage view: layer_index %d out of range for %d layers",
			layerIndex, dimensions[0],
		)
	}

	layerElements := 1

	for _, dimension := range dimensions[1:] {
		layerElements *= dimension
	}

	start := layerIndex * layerElements

	slice, err := storage.Slice(start, layerElements)

	if err != nil {
		return nil, fmt.Errorf("layer storage view: slice layer %d: %w", layerIndex, err)
	}

	return slice.Reshape(dimensions[1:])
}
