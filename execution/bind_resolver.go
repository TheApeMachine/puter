package execution

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type bindResolver struct {
	dispatcher  *dispatcher
	node        *ast.GraphNode
	bind        OperationBind
	output      tensor.Tensor
	outputShape tensor.Shape
	outputDType dtype.DType
}

func runBoundNode(dispatcher *dispatcher, node *ast.GraphNode, bind OperationBind) error {
	resolver := &bindResolver{
		dispatcher: dispatcher,
		node:       node,
		bind:       bind,
	}

	outputShape, err := resolver.resolveOutputShape()

	if err != nil {
		return fmt.Errorf("bind op %q: output shape: %w", node.Op, err)
	}

	outputDType, err := resolver.resolveOutputDType()

	if err != nil {
		return fmt.Errorf("bind op %q: output dtype: %w", node.Op, err)
	}

	resolver.outputShape = outputShape
	resolver.outputDType = outputDType

	if isIntrinsicMethod(bind.Method) {
		output, err := runIntrinsic(resolver)

		if err != nil {
			return fmt.Errorf("bind op %q: %w", node.Op, err)
		}

		dispatcher.values.set(node.ID, output)

		return nil
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return fmt.Errorf("bind op %q: allocate output: %w", node.Op, err)
	}

	resolver.output = output

	configFields, err := resolver.resolveConfigFields()

	if err != nil {
		return err
	}

	args, err := resolver.resolveArgs()

	if err != nil {
		return err
	}

	if err := callRouter(dispatcher.deviceBackend, bind, configFields, args); err != nil {
		return fmt.Errorf("bind op %q: %w", node.Op, err)
	}

	dispatcher.values.set(node.ID, output)

	return nil
}

func (resolver *bindResolver) resolveOutputShape() (tensor.Shape, error) {
	dimensions := make([]int, 0, len(resolver.bind.Output.Shape))

	for index, spec := range resolver.bind.Output.Shape {
		value, err := resolver.resolveArg(spec)

		if err != nil {
			return tensor.Shape{}, fmt.Errorf("dim %d: %w", index, err)
		}

		switch typed := value.(type) {
		case int:
			dimensions = append(dimensions, typed)
		case []int:
			dimensions = append(dimensions, typed...)
		default:
			return tensor.Shape{}, fmt.Errorf("dim %d resolved to %T, expected int or []int", index, value)
		}
	}

	return tensor.NewShape(dimensions)
}

func (resolver *bindResolver) resolveOutputDType() (dtype.DType, error) {
	if resolver.bind.Output.DType == "" {
		return dtype.Float32, nil
	}

	if strings.Contains(resolver.bind.Output.DType, ".") {
		value, err := resolver.resolveArg(asset.BindArg{Ref: resolver.bind.Output.DType})

		if err != nil {
			return dtype.Invalid, err
		}

		format, ok := value.(dtype.DType)

		if !ok {
			return dtype.Invalid, fmt.Errorf("%q resolved to %T, expected dtype.DType", resolver.bind.Output.DType, value)
		}

		return format, nil
	}

	return dtype.Parse(resolver.bind.Output.DType)
}

func (resolver *bindResolver) allocateOutput() (tensor.Tensor, error) {
	byteCount, err := resolver.outputDType.BytesFor(resolver.outputShape.Len())

	if err != nil {
		return nil, err
	}

	output, err := resolver.dispatcher.allocateOutput(
		resolver.node,
		resolver.outputShape,
		resolver.outputDType,
		byteCount,
	)

	if err != nil {
		return nil, err
	}

	if !output.Shape().Equal(resolver.outputShape) {
		return nil, fmt.Errorf("shape %v does not match planned %v", output.Shape().Dims(), resolver.outputShape.Dims())
	}

	if output.DType() != resolver.outputDType {
		return nil, fmt.Errorf("dtype %s does not match planned %s", output.DType(), resolver.outputDType)
	}

	return output, nil
}

func (resolver *bindResolver) resolveConfigFields() (map[string]any, error) {
	fields := make(map[string]any, len(resolver.bind.ConfigFields))

	for fieldName, spec := range resolver.bind.ConfigFields {
		value, err := resolver.resolveArg(spec)

		if err != nil {
			return nil, fmt.Errorf("bind op %q: config field %q: %w", resolver.node.Op, fieldName, err)
		}

		fields[fieldName] = value
	}

	return fields, nil
}

func (resolver *bindResolver) resolveArgs() ([]any, error) {
	args := make([]any, 0, len(resolver.bind.Args))

	for index, spec := range resolver.bind.Args {
		value, err := resolver.resolveArg(spec)

		if err != nil {
			return nil, fmt.Errorf("bind op %q: arg %d: %w", resolver.node.Op, index, err)
		}

		args = append(args, value)
	}

	return args, nil
}

func (resolver *bindResolver) resolveArg(spec asset.BindArg) (any, error) {
	value, err := resolver.resolveRaw(spec)

	if err != nil {
		return nil, err
	}

	return resolver.applyTransforms(value, spec)
}

func (resolver *bindResolver) resolveRaw(spec asset.BindArg) (any, error) {
	if spec.Ref == "" {
		if spec.Value == nil {
			return nil, fmt.Errorf("empty bind arg has no value")
		}

		return scalarInt(spec.Value)
	}

	parts := strings.Split(spec.Ref, ".")

	switch parts[0] {
	case "nil":
		return unsafeNilPointer, nil
	case "input":
		return resolver.resolveInputRef(parts)
	case "output":
		return resolver.resolveOutputRef(parts)
	case "weight":
		return resolver.resolveWeightRef(parts)
	case "config":
		return resolver.resolveConfigRef(parts)
	default:
		return nil, fmt.Errorf("unknown bind ref %q", spec.Ref)
	}
}

func (resolver *bindResolver) resolveInputRef(parts []string) (any, error) {
	if len(parts) < 2 {
		return nil, fmt.Errorf("input ref requires a source name")
	}

	inputTensor, err := resolver.resolveInputTensor(parts[1])

	if err != nil {
		return nil, err
	}

	if len(parts) == 2 {
		return inputTensor, nil
	}

	return tensorProperty(inputTensor, strings.Join(parts[2:], "."))
}

func (resolver *bindResolver) resolveOutputRef(parts []string) (any, error) {
	if len(parts) < 3 {
		return nil, fmt.Errorf("output ref requires a port name and property")
	}

	switch strings.Join(parts[2:], ".") {
	case "pointer":
		if resolver.output == nil {
			return nil, fmt.Errorf("output pointer requested before allocation")
		}

		pointer, _, err := pointerOf(resolver.output)

		return pointer, err
	case "shape":
		return resolver.outputShape.Dims(), nil
	case "dtype":
		return resolver.outputDType, nil
	default:
		return nil, fmt.Errorf("unknown output property %q", strings.Join(parts[2:], "."))
	}
}

func (resolver *bindResolver) resolveWeightRef(parts []string) (any, error) {
	transposed := len(parts) >= 2 && parts[1] == "transposed"
	propertyIndex := 1

	if transposed {
		propertyIndex = 2
	}

	if len(parts) <= propertyIndex {
		return nil, fmt.Errorf("weight ref requires a property")
	}

	weightTensor, err := resolver.resolveWeightTensor(transposed)

	if err != nil {
		return nil, err
	}

	return tensorProperty(weightTensor, strings.Join(parts[propertyIndex:], "."))
}

func (resolver *bindResolver) resolveConfigRef(parts []string) (any, error) {
	if len(parts) != 3 {
		return nil, fmt.Errorf("config ref must be config.<name>.<type>, got %q", strings.Join(parts, "."))
	}

	switch parts[2] {
	case "int":
		return configInt(resolver.node, parts[1], 0), nil
	case "float":
		return float32(configFloat(resolver.node, parts[1], 0)), nil
	case "bool":
		return configBool(resolver.node, parts[1], false), nil
	default:
		return nil, fmt.Errorf("unknown config type %q", parts[2])
	}
}

func tensorProperty(input tensor.Tensor, property string) (any, error) {
	switch property {
	case "pointer":
		pointer, _, err := pointerOf(input)

		return pointer, err
	case "shape":
		return input.Shape().Dims(), nil
	case "len":
		return input.Len(), nil
	case "dtype":
		return input.DType(), nil
	default:
		return nil, fmt.Errorf("unknown tensor property %q", property)
	}
}
