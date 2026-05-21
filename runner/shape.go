package runner

import (
	"fmt"
	"strconv"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func outputShapeForNode(
	node *ir.Node,
	kernel string,
	tensorWorkspace *workspace,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
) (tensor.Shape, error) {
	switch kernel {
	case "embedding_lookup":
		return embeddingOutputShape(
			node,
			tensorWorkspace,
			checkpointPath,
			weights,
			bindings,
		)
	case "linear":
		return linearOutputShape(node, tensorWorkspace)
	case "view_as_heads":
		return viewAsHeadsOutputShape(node, tensorWorkspace)
	case "merge_heads":
		return mergeHeadsOutputShape(node, tensorWorkspace)
	case "rmsnorm", "layernorm", "rope", "add", "mul", "swiglu",
		"relu", "gelu", "tanh", "sigmoid", "swish", "selu", "leaky_relu",
		"slice", "concat", "transpose",
		"reshape", "dropout", "softmax":
		return primaryFloatInputShape(node, tensorWorkspace)
	default:
		return node.Shape(), nil
	}
}

func linearOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)

	if err != nil {
		return tensor.Shape{}, err
	}

	outFeatures, err := nodeIntAttribute(node, "out_features")

	if err != nil {
		return tensor.Shape{}, err
	}

	dims := append([]int(nil), inputShape.Dims()...)

	if len(dims) == 0 {
		return tensor.Shape{}, fmt.Errorf("runner: linear node %q has empty input shape", node.ID())
	}

	dims[len(dims)-1] = outFeatures

	return tensor.NewShape(dims)
}

func viewAsHeadsOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)

	if err != nil {
		return tensor.Shape{}, err
	}

	numHeads, err := nodeIntAttribute(node, "num_heads")

	if err != nil {
		return tensor.Shape{}, err
	}

	inputDims := inputShape.Dims()

	if len(inputDims) != 3 {
		return tensor.Shape{}, fmt.Errorf(
			"runner: view_as_heads node %q expects [batch, seq, hidden], got %d dims",
			node.ID(),
			len(inputDims),
		)
	}

	hiddenSize := inputDims[2]

	if hiddenSize%numHeads != 0 {
		return tensor.Shape{}, fmt.Errorf(
			"runner: view_as_heads node %q hidden size %d is not divisible by num_heads %d",
			node.ID(),
			hiddenSize,
			numHeads,
		)
	}

	headDim := hiddenSize / numHeads

	return tensor.NewShape([]int{inputDims[0], inputDims[1], numHeads, headDim})
}

func mergeHeadsOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)

	if err != nil {
		return tensor.Shape{}, err
	}

	inputDims := inputShape.Dims()

	if len(inputDims) != 4 {
		return tensor.Shape{}, fmt.Errorf(
			"runner: merge_heads node %q expects [batch, seq, heads, head_dim], got %d dims",
			node.ID(),
			len(inputDims),
		)
	}

	hiddenSize := inputDims[2] * inputDims[3]

	return tensor.NewShape([]int{inputDims[0], inputDims[1], hiddenSize})
}

func nodeIntAttribute(node *ir.Node, key string) (int, error) {
	attribute := node.Attribute(key)

	if attribute.Kind == ir.AttributeInt {
		parsed, err := strconv.ParseInt(attribute.Value, 10, 64)

		if err != nil {
			return 0, err
		}

		return int(parsed), nil
	}

	if metadata := node.Metadata(); metadata != nil {
		if raw, ok := metadata[key]; ok {
			switch typed := raw.(type) {
			case int:
				return typed, nil
			case int64:
				return int(typed), nil
			case float64:
				return int(typed), nil
			}
		}
	}

	return 0, fmt.Errorf("runner: node %q missing integer attribute %q", node.ID(), key)
}

func primaryFloatInputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok || !value.DType().IsFloat() {
			continue
		}

		return value.Shape(), nil
	}

	return node.Shape(), nil
}

func embeddingOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
) (tensor.Shape, error) {
	indices, err := embeddingIndicesTensor(node, tensorWorkspace)

	if err != nil {
		return tensor.Shape{}, err
	}

	hiddenSize, err := embeddingHiddenSize(node, checkpointPath, weights, bindings)

	if err != nil {
		return tensor.Shape{}, err
	}

	indexDims := indices.Shape().Dims()

	switch len(indexDims) {
	case 1:
		return tensor.NewShape([]int{1, indexDims[0], hiddenSize})
	case 2:
		return tensor.NewShape([]int{indexDims[0], indexDims[1], hiddenSize})
	default:
		return tensor.Shape{}, fmt.Errorf(
			"runner: embedding node %q expects 1D or 2D indices, got %d dims",
			node.ID(),
			len(indexDims),
		)
	}
}

func embeddingIndicesTensor(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Tensor, error) {
	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok {
			continue
		}

		if value.DType() == dtype.Int32 {
			return value, nil
		}
	}

	return nil, fmt.Errorf("runner: embedding node %q is missing int32 indices", node.ID())
}

func embeddingHiddenSize(
	node *ir.Node,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
) (int, error) {
	weightName := bindings.weightTensorName(node.ID())

	if weightName == "" {
		weightName = weightTensorName(node)
	}

	if weightName != "" && checkpointPath != "" && weights != nil {
		weight, err := weights.Tensor(checkpointPath, weightName)

		if err == nil {
			weightDims := weight.Shape().Dims()

			if len(weightDims) == 2 {
				return weightDims[1], nil
			}
		}
	}

	if rawShape, ok := node.Metadata()["weight_shape"].([]int64); ok && len(rawShape) == 2 {
		return int(rawShape[1]), nil
	}

	if rawShape, ok := node.Metadata()["weight_shape"].([]int); ok && len(rawShape) == 2 {
		return rawShape[1], nil
	}

	return 0, fmt.Errorf("runner: embedding node %q is missing weight hidden size", node.ID())
}
