package runner

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func appendKernelAttributes(
	memory tensor.Backend,
	node *ir.Node,
	kernel string,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("runner: node %q has no dispatch arguments", node.ID())
	}

	switch kernel {
	case "view_as_heads":
		return appendInt32AttributeBeforeOutput(memory, node, "num_heads", args)
	case "timestep":
		return appendTimestepAttributes(memory, node, args)
	case "slice":
		return appendSliceAttributes(memory, node, args)
	case "transpose":
		return appendTransposeAttributes(memory, node, args)
	case "page_write", "page_gather":
		return appendInt32AttributeBeforeOutput(memory, node, "page_size", args)
	default:
		return args, nil
	}
}

func appendTransposeAttributes(
	memory tensor.Backend,
	node *ir.Node,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	if len(args) < 2 {
		return nil, fmt.Errorf("runner: transpose node %q has no input tensor", node.ID())
	}

	dims := args[0].Shape().Dims()
	dim0, err := nodeIntAttribute(node, "dim0")

	if err != nil {
		return nil, err
	}

	dim1, err := nodeIntAttribute(node, "dim1")

	if err != nil {
		return nil, err
	}

	if dim0 < 0 {
		dim0 += len(dims)
	}

	if dim1 < 0 {
		dim1 += len(dims)
	}

	if dim0 < 0 || dim0 >= len(dims) || dim1 < 0 || dim1 >= len(dims) {
		return nil, fmt.Errorf("runner: transpose node %q dims out of range", node.ID())
	}

	permutation := make([]int32, len(dims))

	for index := range permutation {
		permutation[index] = int32(index)
	}

	permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]

	permutationTensor, err := uploadInt32Slice(memory, permutation)
	if err != nil {
		return nil, err
	}

	output := args[len(args)-1]
	inputs := args[:len(args)-1]

	return append(append(inputs, permutationTensor), output), nil
}

func appendSliceAttributes(
	memory tensor.Backend,
	node *ir.Node,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	for _, attributeName := range []string{"dim", "start", "end"} {
		var err error
		args, err = appendInt32AttributeBeforeOutput(memory, node, attributeName, args)

		if err != nil {
			return nil, err
		}
	}

	return args, nil
}

func nodeFloat32Attribute(node *ir.Node, key string, fallback float32) float32 {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeFloat {
		parsed, err := strconv.ParseFloat(attribute.Value, 32)
		if err == nil {
			return float32(parsed)
		}
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			return float32FromAny(raw, fallback)
		}
	}

	return fallback
}

func nodeBoolAttribute(node *ir.Node, key string, fallback bool) bool {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeBool {
		return attribute.Value == "true"
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			if typed, ok := raw.(bool); ok {
				return typed
			}
		}
	}

	return fallback
}

func float32FromAny(value any, fallback float32) float32 {
	switch typed := value.(type) {
	case float32:
		return typed
	case float64:
		return float32(typed)
	case int:
		return float32(typed)
	case int64:
		return float32(typed)
	default:
		return fallback
	}
}

func appendTimestepAttributes(
	memory tensor.Backend,
	node *ir.Node,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	maxPeriod := nodeFloat32Attribute(node, "max_period", 10000)
	downscale := nodeFloat32Attribute(node, "downscale_freq_shift", 0)
	flip := nodeBoolAttribute(node, "flip_sin_to_cos", false)

	maxPeriodTensor, err := uploadFloat32Scalar(memory, maxPeriod)
	if err != nil {
		return nil, err
	}

	downscaleTensor, err := uploadFloat32Scalar(memory, downscale)
	if err != nil {
		return nil, err
	}

	flipValue := 0
	if flip {
		flipValue = 1
	}

	flipTensor, err := uploadInt32Scalar(memory, flipValue)
	if err != nil {
		return nil, err
	}

	output := args[len(args)-1]
	inputs := args[:len(args)-1]

	return append(append(inputs, maxPeriodTensor, downscaleTensor, flipTensor), output), nil
}

func appendInt32AttributeBeforeOutput(
	memory tensor.Backend,
	node *ir.Node,
	attributeName string,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	value, err := nodeIntAttribute(node, attributeName)

	if err != nil {
		return nil, fmt.Errorf("runner: node %q: %w", node.ID(), err)
	}

	scalar, err := uploadInt32Scalar(memory, value)

	if err != nil {
		return nil, err
	}

	output := args[len(args)-1]
	inputs := args[:len(args)-1]

	return append(append(inputs, scalar), output), nil
}

func uploadInt32Scalar(memory tensor.Backend, value int) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{1})

	if err != nil {
		return nil, err
	}

	buffer := make([]byte, 4)
	binary.LittleEndian.PutUint32(buffer, uint32(value))

	return memory.Upload(shape, dtype.Int32, buffer)
}

func uploadFloat32Scalar(memory tensor.Backend, value float32) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{1})

	if err != nil {
		return nil, err
	}

	buffer := make([]byte, 4)
	binary.LittleEndian.PutUint32(buffer, math.Float32bits(value))

	return memory.Upload(shape, dtype.Float32, buffer)
}
