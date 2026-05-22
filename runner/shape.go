package runner

import (
	"fmt"
	"strconv"
	"strings"

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
	case "timestep":
		return timestepOutputShape(node, tensorWorkspace)
	case "linear":
		return linearOutputShape(node, tensorWorkspace)
	case "view_as_heads":
		return viewAsHeadsOutputShape(node, tensorWorkspace)
	case "merge_heads":
		return mergeHeadsOutputShape(node, tensorWorkspace)
	case "grouped_query_attention", "multi_head_attention", "flash_attention", "attention":
		return primaryFloatInputShape(node, tensorWorkspace)
	case "add", "mul":
		return binaryFloatOutputShape(node, tensorWorkspace)
	case "concat":
		return concatOutputShape(node, tensorWorkspace)
	case "swiglu":
		return swiGLUOutputShape(node, tensorWorkspace)
	case "slice":
		return sliceOutputShape(node, tensorWorkspace)
	case "transpose":
		return transposeOutputShape(node, tensorWorkspace)
	case "reshape":
		return reshapeOutputShape(node, tensorWorkspace)
	case "conv2d":
		return conv2DOutputShape(node, tensorWorkspace)
	case "upsample_nearest2d":
		return upsampleNearest2DOutputShape(node, tensorWorkspace)
	case "rmsnorm", "layernorm", "groupnorm", "instancenorm", "batchnorm_eval",
		"adaptive_rmsnorm", "batchnorm_denorm",
		"modulated_layernorm", "gated_residual", "rope", "multi_axis_rope",
		"relu", "gelu", "tanh", "sigmoid", "swish", "selu", "leaky_relu",
		"dropout", "softmax":
		return primaryFloatInputShape(node, tensorWorkspace)
	default:
		return node.Shape(), nil
	}
}

func upsampleNearest2DOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)
	if err != nil {
		return tensor.Shape{}, err
	}

	dims := append([]int(nil), inputShape.Dims()...)
	if len(dims) != 4 {
		return tensor.Shape{}, fmt.Errorf("runner: upsample_nearest2d node %q expects NCHW input", node.ID())
	}

	scaleHeight, err := nodeIntAttribute(node, "scale_h")
	if err != nil {
		scaleHeight = 1
	}

	scaleWidth, err := nodeIntAttribute(node, "scale_w")
	if err != nil {
		scaleWidth = 1
	}

	dims[2] *= scaleHeight
	dims[3] *= scaleWidth

	return tensor.NewShape(dims)
}

func conv2DOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)
	if err != nil {
		return tensor.Shape{}, err
	}

	dims := inputShape.Dims()
	if len(dims) != 4 {
		return tensor.Shape{}, fmt.Errorf("runner: conv2d node %q expects NCHW input", node.ID())
	}

	outChannels, err := nodeIntAttribute(node, "out_channels")
	if err != nil {
		return tensor.Shape{}, err
	}

	kernelHeight, err := nodeIntAttribute(node, "kernel_h")
	if err != nil {
		return tensor.Shape{}, err
	}

	kernelWidth, err := nodeIntAttribute(node, "kernel_w")
	if err != nil {
		return tensor.Shape{}, err
	}

	strideHeight, err := nodeIntAttribute(node, "stride_h")
	if err != nil {
		strideHeight = 1
	}

	strideWidth, err := nodeIntAttribute(node, "stride_w")
	if err != nil {
		strideWidth = 1
	}

	paddingHeight, err := nodeIntAttribute(node, "pad_h")
	if err != nil {
		paddingHeight = 0
	}

	paddingWidth, err := nodeIntAttribute(node, "pad_w")
	if err != nil {
		paddingWidth = 0
	}

	outHeight := (dims[2]+2*paddingHeight-kernelHeight)/strideHeight + 1
	outWidth := (dims[3]+2*paddingWidth-kernelWidth)/strideWidth + 1

	if outHeight <= 0 || outWidth <= 0 {
		return tensor.Shape{}, fmt.Errorf("runner: conv2d node %q has invalid output size", node.ID())
	}

	return tensor.NewShape([]int{dims[0], outChannels, outHeight, outWidth})
}

func reshapeOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	if dims, ok := reshapeDimsFromNode(node); ok {
		return tensor.NewShape(dims)
	}

	return primaryFloatInputShape(node, tensorWorkspace)
}

func reshapeDimsFromNode(node *ir.Node) ([]int, bool) {
	if metadata := node.Metadata(); metadata != nil {
		if dims, ok := intSliceFromAny(metadata["shape"]); ok {
			return dims, true
		}
	}

	attribute := node.Attribute("shape")
	if attribute.Value == "" {
		return nil, false
	}

	value := strings.TrimSpace(attribute.Value)
	value = strings.TrimPrefix(value, "[")
	value = strings.TrimSuffix(value, "]")

	parts := strings.Split(value, ",")
	if len(parts) == 1 {
		parts = strings.Fields(value)
	}
	dims := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		dimension, err := strconv.Atoi(part)
		if err != nil {
			return nil, false
		}

		dims = append(dims, dimension)
	}

	return dims, len(dims) > 0
}

func intSliceFromAny(value any) ([]int, bool) {
	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), true
	case []int64:
		dims := make([]int, len(typed))

		for index, dimension := range typed {
			dims[index] = int(dimension)
		}

		return dims, true
	case []any:
		dims := make([]int, len(typed))

		for index, raw := range typed {
			switch dimension := raw.(type) {
			case int:
				dims[index] = dimension
			case int64:
				dims[index] = int(dimension)
			case float64:
				dims[index] = int(dimension)
			default:
				return nil, false
			}
		}

		return dims, true
	default:
		return nil, false
	}
}

func transposeOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)
	if err != nil {
		return tensor.Shape{}, err
	}

	dims := append([]int(nil), inputShape.Dims()...)
	dim0, err := nodeIntAttribute(node, "dim0")
	if err != nil {
		return tensor.Shape{}, err
	}

	dim1, err := nodeIntAttribute(node, "dim1")
	if err != nil {
		return tensor.Shape{}, err
	}

	if dim0 < 0 {
		dim0 += len(dims)
	}

	if dim1 < 0 {
		dim1 += len(dims)
	}

	if dim0 < 0 || dim0 >= len(dims) || dim1 < 0 || dim1 >= len(dims) {
		return tensor.Shape{}, fmt.Errorf("runner: transpose node %q dims out of range", node.ID())
	}

	dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

	return tensor.NewShape(dims)
}

func sliceOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)
	if err != nil {
		return tensor.Shape{}, err
	}

	dims := append([]int(nil), inputShape.Dims()...)
	axis, err := nodeIntAttribute(node, "dim")
	if err != nil {
		return tensor.Shape{}, err
	}

	if axis < 0 {
		axis += len(dims)
	}

	if axis < 0 || axis >= len(dims) {
		return tensor.Shape{}, fmt.Errorf("runner: slice node %q axis %d out of range", node.ID(), axis)
	}

	start, err := nodeIntAttribute(node, "start")
	if err != nil {
		return tensor.Shape{}, err
	}

	end, err := nodeIntAttribute(node, "end")
	if err != nil {
		return tensor.Shape{}, err
	}

	if end == 0 {
		end = dims[axis]
	}

	if start < 0 {
		start += dims[axis]
	}

	if end < 0 {
		end += dims[axis]
	}

	if start < 0 || end < start || end > dims[axis] {
		return tensor.Shape{}, fmt.Errorf("runner: slice node %q bounds [%d:%d] out of range", node.ID(), start, end)
	}

	dims[axis] = end - start

	return tensor.NewShape(dims)
}

func timestepOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	inputShape, err := primaryFloatInputShape(node, tensorWorkspace)

	if err != nil {
		return tensor.Shape{}, err
	}

	dim, err := nodeIntAttribute(node, "dim")

	if err != nil {
		return tensor.Shape{}, err
	}

	inputDims := inputShape.Dims()

	if len(inputDims) == 0 {
		return tensor.Shape{}, fmt.Errorf("runner: timestep node %q has empty input shape", node.ID())
	}

	return tensor.NewShape([]int{inputShape.Len(), dim})
}

func swiGLUOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	floatShapes := make([]tensor.Shape, 0, len(node.Inputs()))

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok || !value.DType().IsFloat() {
			continue
		}

		floatShapes = append(floatShapes, value.Shape())
	}

	if len(floatShapes) != 1 {
		return primaryFloatInputShape(node, tensorWorkspace)
	}

	dims := append([]int(nil), floatShapes[0].Dims()...)

	if len(dims) == 0 || dims[len(dims)-1]%2 != 0 {
		return tensor.Shape{}, fmt.Errorf("runner: packed swiglu node %q requires even final dimension", node.ID())
	}

	dims[len(dims)-1] /= 2

	return tensor.NewShape(dims)
}

func concatOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	floatShapes := make([]tensor.Shape, 0, len(node.Inputs()))

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok || !value.DType().IsFloat() {
			continue
		}

		floatShapes = append(floatShapes, value.Shape())
	}

	if len(floatShapes) == 0 {
		return node.Shape(), nil
	}

	dims := append([]int(nil), floatShapes[0].Dims()...)
	axis, err := nodeIntAttribute(node, "dim")

	if err != nil {
		axis, err = nodeIntAttribute(node, "axis")
	}

	if err != nil {
		axis = len(dims) - 1
	}

	if axis < 0 {
		axis += len(dims)
	}

	if axis < 0 || axis >= len(dims) {
		return tensor.Shape{}, fmt.Errorf("runner: concat node %q axis %d out of range", node.ID(), axis)
	}

	for index := 1; index < len(floatShapes); index++ {
		inputDims := floatShapes[index].Dims()

		if len(inputDims) != len(dims) {
			return tensor.Shape{}, fmt.Errorf("runner: concat node %q rank mismatch", node.ID())
		}

		for dimensionIndex, dimension := range inputDims {
			if dimensionIndex == axis {
				dims[axis] += dimension
				continue
			}

			if dims[dimensionIndex] != dimension {
				return tensor.Shape{}, fmt.Errorf(
					"runner: concat node %q dim %d shape %d != %d",
					node.ID(),
					dimensionIndex,
					dims[dimensionIndex],
					dimension,
				)
			}
		}
	}

	return tensor.NewShape(dims)
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

func binaryFloatOutputShape(
	node *ir.Node,
	tensorWorkspace *workspace,
) (tensor.Shape, error) {
	floatShapes := make([]tensor.Shape, 0, len(node.Inputs()))

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok || !value.DType().IsFloat() {
			continue
		}

		floatShapes = append(floatShapes, value.Shape())
	}

	if len(floatShapes) == 0 {
		return node.Shape(), nil
	}

	referenceShape := floatShapes[0]

	for index := 1; index < len(floatShapes); index++ {
		if !referenceShape.Equal(floatShapes[index]) {
			return tensor.Shape{}, fmt.Errorf(
				"runner: node %q input shape %v != %v",
				node.ID(),
				referenceShape.Dims(),
				floatShapes[index].Dims(),
			)
		}
	}

	return referenceShape, nil
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

	if rawShape, ok := node.Metadata()["weight_shape"].([]int64); ok && len(rawShape) == 2 {
		return int(rawShape[1]), nil
	}

	if rawShape, ok := node.Metadata()["weight_shape"].([]int); ok && len(rawShape) == 2 {
		return rawShape[1], nil
	}

	weightPath := weightFilePath(node, checkpointPath)

	if weightName != "" && weightPath != "" && weights != nil {
		weight, err := weights.Tensor(weightPath, weightName)

		if err == nil {
			weightDims := weight.Shape().Dims()

			if len(weightDims) == 2 {
				return weightDims[1], nil
			}
		}
	}

	return 0, fmt.Errorf("runner: embedding node %q is missing weight hidden size", node.ID())
}
