package execution

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/tensor"
)

func (resolver *bindResolver) resolveWeightTensor(transposed bool, bias bool) (tensor.Tensor, error) {
	if resolver.node.Weights == nil || resolver.node.Weights.TensorName == "" {
		return nil, fmt.Errorf("bind: node %q requires a weight binding", resolver.node.ID)
	}

	if bias {
		return resolver.resolveBiasTensor()
	}

	if resolver.node.Weights.Slice != nil {
		return resolver.resolveSlicedWeightTensor(transposed)
	}

	if !transposed {
		return resolver.dispatcher.weights.Lookup(resolver.node.Weights.TensorName)
	}

	transposedStore, ok := resolver.dispatcher.weights.(TransposedLookup)

	if !ok {
		return nil, fmt.Errorf(
			"weight store does not implement TransposedLookup for %q",
			resolver.node.Weights.TensorName,
		)
	}

	return transposedStore.LookupTransposed(resolver.node.Weights.TensorName)
}

func (resolver *bindResolver) resolveSlicedWeightTensor(transposed bool) (tensor.Tensor, error) {
	axis, start, end, err := resolver.weightSliceRange()

	if err != nil {
		return nil, err
	}

	if transposed {
		transposedStore, ok := resolver.dispatcher.weights.(TransposedSliceLookup)

		if !ok {
			return nil, fmt.Errorf(
				"weight store does not implement TransposedSliceLookup for %q",
				resolver.node.Weights.TensorName,
			)
		}

		return transposedStore.LookupTransposedSlice(
			resolver.node.Weights.TensorName,
			axis,
			start,
			end,
		)
	}

	slicedStore, ok := resolver.dispatcher.weights.(SliceLookup)

	if !ok {
		return nil, fmt.Errorf(
			"weight store does not implement SliceLookup for %q",
			resolver.node.Weights.TensorName,
		)
	}

	return slicedStore.LookupSlice(resolver.node.Weights.TensorName, axis, start, end)
}

func (resolver *bindResolver) resolveBiasTensor() (tensor.Tensor, error) {
	if resolver.node.Weights.BiasName == "" {
		return resolver.resolveDefaultBiasTensor()
	}

	if resolver.node.Weights.Slice != nil && resolver.node.Weights.Slice.Axis == "output" {
		return resolver.resolveSlicedBiasTensor()
	}

	return resolver.dispatcher.weights.Lookup(resolver.node.Weights.BiasName)
}

func (resolver *bindResolver) resolveDefaultBiasTensor() (tensor.Tensor, error) {
	weightName := resolver.node.Weights.TensorName

	if !strings.HasSuffix(weightName, ".weight") {
		return nil, fmt.Errorf("bind: node %q requires a bias binding", resolver.node.ID)
	}

	biasName := strings.TrimSuffix(weightName, ".weight") + ".bias"
	resident, err := resolver.dispatcher.weights.Lookup(biasName)

	if err != nil {
		return nil, fmt.Errorf("bind: node %q requires bias tensor %q: %w", resolver.node.ID, biasName, err)
	}

	return resident, nil
}

func (resolver *bindResolver) resolveSlicedBiasTensor() (tensor.Tensor, error) {
	axis, start, end, err := resolver.weightSliceRange()

	if err != nil {
		return nil, err
	}

	slicedStore, ok := resolver.dispatcher.weights.(SliceLookup)

	if !ok {
		return nil, fmt.Errorf(
			"weight store does not implement SliceLookup for %q",
			resolver.node.Weights.BiasName,
		)
	}

	return slicedStore.LookupSlice(resolver.node.Weights.BiasName, axis, start, end)
}

func (resolver *bindResolver) weightSliceRange() (string, int64, int64, error) {
	selection := resolver.node.Weights.Slice

	if selection == nil {
		return "", 0, 0, fmt.Errorf("bind: node %q has no weight slice", resolver.node.ID)
	}

	if selection.Axis == "" {
		return "", 0, 0, fmt.Errorf("bind: node %q weight slice axis is required", resolver.node.ID)
	}

	start := selection.Start
	end := selection.End

	if end > start {
		return selection.Axis, start, end, nil
	}

	length, err := resolver.weightSliceLength(selection.Axis)

	if err != nil {
		return "", 0, 0, err
	}

	return selection.Axis, start, start + int64(length), nil
}

func (resolver *bindResolver) weightSliceLength(axis string) (int, error) {
	var key string

	switch axis {
	case "output":
		key = "out_features"
	case "input":
		key = "in_features"
	default:
		return 0, fmt.Errorf("bind: node %q unsupported weight slice axis %q", resolver.node.ID, axis)
	}

	length := configInt(resolver.node, key, resolver.defaultConfigInt(key))

	if length <= 0 {
		return 0, fmt.Errorf("bind: node %q cannot infer weight slice length from %s", resolver.node.ID, key)
	}

	return length, nil
}
