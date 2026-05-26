package execution

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"unsafe"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (resolver *bindResolver) applyTransforms(value any, spec asset.BindArg) (any, error) {
	switch typed := value.(type) {
	case []int:
		return resolver.applyShapeTransforms(typed, spec)
	case int:
		return resolver.applyIntTransforms(typed, spec)
	default:
		return value, nil
	}
}

func (resolver *bindResolver) applyShapeTransforms(dimensions []int, spec asset.BindArg) (any, error) {
	dims := append([]int(nil), dimensions...)

	if spec.DropTail > 0 {
		if spec.DropTail > len(dims) {
			return nil, fmt.Errorf("drop_tail %d exceeds shape %v", spec.DropTail, dimensions)
		}

		dims = dims[:len(dims)-spec.DropTail]
	}

	if spec.ProductTail > 0 {
		if spec.ProductTail > len(dims) {
			return nil, fmt.Errorf("product_tail %d exceeds shape %v", spec.ProductTail, dimensions)
		}

		return resolver.applyIntTransforms(productInts(dims[len(dims)-spec.ProductTail:]), spec)
	}

	if spec.Dim != nil {
		dimension, err := shapeDim(dims, *spec.Dim)

		if err != nil {
			return nil, err
		}

		return resolver.applyIntTransforms(dimension, spec)
	}

	if spec.Product {
		return resolver.applyIntTransforms(productInts(dims), spec)
	}

	return dims, nil
}

func (resolver *bindResolver) applyIntTransforms(value int, spec asset.BindArg) (int, error) {
	divisor := spec.Divisor

	if spec.DivisorRef != "" {
		raw, err := resolver.resolveArg(asset.BindArg{Ref: spec.DivisorRef})

		if err != nil {
			return 0, err
		}

		resolvedDivisor, ok := raw.(int)

		if !ok {
			return 0, fmt.Errorf("divisor ref %q resolved to %T, expected int", spec.DivisorRef, raw)
		}

		divisor = resolvedDivisor
	}

	if divisor == 0 {
		return value, nil
	}

	if value%divisor != 0 {
		return 0, fmt.Errorf("%d is not divisible by %d", value, divisor)
	}

	return value / divisor, nil
}

func (resolver *bindResolver) resolveInputTensor(source string) (tensor.Tensor, error) {
	inputIndex, err := resolver.inputIndex(source)

	if err != nil {
		return nil, err
	}

	inputName := resolver.node.Inputs[inputIndex]
	raw, ok := resolver.dispatcher.values.get(inputName)

	if ok {
		inputTensor, err := resolver.tensorFromValue(raw)

		if err != nil {
			return nil, fmt.Errorf("input %q: %w", inputName, err)
		}

		resolver.dispatcher.values.set(inputName, inputTensor)

		return inputTensor, nil
	}

	if resolver.dispatcher.workspaces != nil {
		inputs, ok := resolver.dispatcher.workspaces.InputsFor(
			resolver.dispatcher.graphName,
			resolver.node.ID,
		)

		if ok && inputIndex < len(inputs) && inputs[inputIndex] != nil {
			return inputs[inputIndex], nil
		}
	}

	return nil, fmt.Errorf("execution: value %q not found", inputName)
}

func (resolver *bindResolver) inputIndex(source string) (int, error) {
	if source == "" && len(resolver.node.Inputs) == 1 {
		return 0, nil
	}

	if inputIndex, ok := parseNonNegativeInt(source); ok {
		return resolver.checkInputIndex(inputIndex)
	}

	for inputIndex, name := range resolver.bind.InputNames {
		if name != source {
			continue
		}

		return resolver.checkInputIndex(inputIndex)
	}

	return 0, fmt.Errorf("bind: input source %q is not declared for op %q", source, resolver.node.Op)
}

func (resolver *bindResolver) checkInputIndex(inputIndex int) (int, error) {
	if inputIndex < 0 || inputIndex >= len(resolver.node.Inputs) {
		return 0, fmt.Errorf(
			"bind: input index %d out of range for node %q with %d inputs",
			inputIndex, resolver.node.ID, len(resolver.node.Inputs),
		)
	}

	return inputIndex, nil
}

func (resolver *bindResolver) tensorFromValue(value any) (tensor.Tensor, error) {
	switch typed := value.(type) {
	case tensor.Tensor:
		return typed, nil
	case []int:
		return resolver.uploadIntSlice(typed)
	case []int32:
		return resolver.uploadInt32Slice(typed)
	case []int64:
		converted := make([]int32, len(typed))

		for index, value := range typed {
			if value < minInt32 || value > maxInt32 {
				return nil, fmt.Errorf("int64 token value %d overflows int32", value)
			}

			converted[index] = int32(value)
		}

		return resolver.uploadInt32Slice(converted)
	case []float32:
		return resolver.uploadFloat32Slice(typed)
	default:
		return nil, fmt.Errorf("has type %T, expected tensor.Tensor or host slice", value)
	}
}

func (resolver *bindResolver) uploadIntSlice(values []int) (tensor.Tensor, error) {
	converted := make([]int32, len(values))

	for index, value := range values {
		if value < minInt32 || value > maxInt32 {
			return nil, fmt.Errorf("token value %d overflows int32", value)
		}

		converted[index] = int32(value)
	}

	return resolver.uploadInt32Slice(converted)
}

func (resolver *bindResolver) uploadInt32Slice(values []int32) (tensor.Tensor, error) {
	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], uint32(value))
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	return resolver.dispatcher.memory.Upload(shape, dtype.Int32, buffer)
}

func (resolver *bindResolver) uploadFloat32Slice(values []float32) (tensor.Tensor, error) {
	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	return resolver.dispatcher.memory.Upload(shape, dtype.Float32, buffer)
}

func (resolver *bindResolver) resolveWeightTensor(transposed bool) (tensor.Tensor, error) {
	if resolver.node.Weights == nil || resolver.node.Weights.TensorName == "" {
		return nil, fmt.Errorf("bind: node %q requires a weight binding", resolver.node.ID)
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

func shapeDim(dimensions []int, dimensionIndex int) (int, error) {
	if dimensionIndex < 0 {
		dimensionIndex = len(dimensions) + dimensionIndex
	}

	if dimensionIndex < 0 || dimensionIndex >= len(dimensions) {
		return 0, fmt.Errorf("dim %d out of range for shape %v", dimensionIndex, dimensions)
	}

	return dimensions[dimensionIndex], nil
}

func productInts(values []int) int {
	product := 1

	for _, value := range values {
		product *= value
	}

	return product
}

func parseNonNegativeInt(text string) (int, bool) {
	value, err := strconv.Atoi(text)

	if err != nil || value < 0 {
		return 0, false
	}

	return value, true
}

func scalarInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	default:
		return 0, fmt.Errorf("literal %T is not supported", value)
	}
}

var unsafeNilPointer unsafe.Pointer

const (
	minInt32 = -1 << 31
	maxInt32 = 1<<31 - 1
)
