package execution

import (
	"encoding/binary"
	"fmt"
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
	inputSlots  []int
	outputSlot  int
}

func runBoundNode(dispatcher *dispatcher, node *ast.GraphNode, bind OperationBind) error {
	return runBoundNodeWithSlots(dispatcher, node, bind, nil, -1)
}

func runBoundNodeWithSlots(
	dispatcher *dispatcher,
	node *ast.GraphNode,
	bind OperationBind,
	inputSlots []int,
	outputSlot int,
) error {
	resolver := &bindResolver{
		dispatcher: dispatcher,
		node:       node,
		bind:       bind,
		inputSlots: inputSlots,
		outputSlot: outputSlot,
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

		resolver.storeOutput(output)

		return nil
	}

	if isPageIntrinsicMethod(bind.Method) {
		if err := runPageIntrinsic(resolver); err != nil {
			return fmt.Errorf("bind op %q: %w", node.Op, err)
		}

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

	if bind.Method == "RoPE" && len(node.Inputs) >= 2 {
		positionValue, err := resolver.resolveInputTensor(node.Inputs[1])

		if err != nil {
			return fmt.Errorf("bind op %q: rope position: %w", node.Op, err)
		}

		startPosition, err := scalarInt32Tensor(positionValue)

		if err != nil {
			return fmt.Errorf("bind op %q: rope position: %w", node.Op, err)
		}

		configFields["StartPosition"] = int(startPosition)
	}

	args, err := resolver.resolveArgs()

	if err != nil {
		return err
	}

	call, err := bind.deviceCall()

	if err != nil {
		return fmt.Errorf("bind op %q: %w", node.Op, err)
	}

	if err := call(dispatcher.deviceBackend, configFields, args); err != nil {
		return fmt.Errorf("bind op %q: %w", node.Op, err)
	}

	resolver.storeOutput(output)

	return nil
}

func (resolver *bindResolver) storeOutput(value any) {
	if resolver.dispatcher.values.hasSlot(resolver.outputSlot) {
		resolver.dispatcher.values.setSlot(resolver.outputSlot, value)
		return
	}

	resolver.dispatcher.values.set(resolver.node.ID, value)
}

func scalarInt32Tensor(value tensor.Tensor) (int32, error) {
	if value.DType() != dtype.Int32 {
		return 0, fmt.Errorf("expected int32 tensor, got %s", value.DType())
	}

	if value.Location() != tensor.Host {
		dataType, rawBytes, err := value.RawBytes()

		if err != nil {
			return 0, err
		}

		if dataType != dtype.Int32 {
			return 0, fmt.Errorf("expected int32 tensor bytes, got %s", dataType)
		}

		if len(rawBytes) != 4 {
			return 0, fmt.Errorf("expected scalar int32 tensor bytes, got %d bytes", len(rawBytes))
		}

		return int32(binary.LittleEndian.Uint32(rawBytes)), nil
	}

	values, err := value.Int32Native()

	if err != nil {
		return 0, err
	}

	if len(values) != 1 {
		return 0, fmt.Errorf("expected scalar int32 tensor, got len %d", len(values))
	}

	return values[0], nil
}

func (resolver *bindResolver) resolveOutputShape() (tensor.Shape, error) {
	if resolver.bind.Method == "shape.concat" {
		return resolver.resolveConcatOutputShape()
	}

	if resolver.bind.Method == "shape.last_token" {
		return resolver.resolveLastTokenOutputShape()
	}

	if resolver.bind.Method == "shape.slice" {
		return resolver.resolveSliceOutputShape()
	}

	if resolver.bind.Method == "shape.transpose" {
		return resolver.resolveTransposeOutputShape()
	}

	if resolver.bind.Method == "shape.upsample_nearest2d" {
		return resolver.resolveUpsampleNearest2DOutputShape()
	}

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

	return resolver.outputView(output, byteCount)
}

func (resolver *bindResolver) outputView(output tensor.Tensor, byteCount int) (tensor.Tensor, error) {
	if output.DType() != resolver.outputDType {
		return nil, fmt.Errorf("dtype %s does not match planned %s", output.DType(), resolver.outputDType)
	}

	if output.Shape().Equal(resolver.outputShape) {
		return output, nil
	}

	if output.Bytes() < byteCount {
		return nil, fmt.Errorf(
			"workspace output %v has %d bytes, need %d bytes for live shape %v",
			output.Shape().Dims(),
			output.Bytes(),
			byteCount,
			resolver.outputShape.Dims(),
		)
	}

	view, err := output.Slice(0, resolver.outputShape.Len())

	if err != nil {
		return nil, fmt.Errorf("slice workspace output %v to %v: %w", output.Shape().Dims(), resolver.outputShape.Dims(), err)
	}

	reshaped, err := view.Reshape(resolver.outputShape.Dims())

	if err != nil {
		return nil, fmt.Errorf("reshape workspace output %v to %v: %w", output.Shape().Dims(), resolver.outputShape.Dims(), err)
	}

	return reshaped, nil
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
	case "launch":
		return resolver.resolveLaunchRef(parts[1:])
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

func (resolver *bindResolver) resolveLaunchRef(parts []string) (any, error) {
	if len(parts) != 2 || parts[1] != "int" {
		return nil, fmt.Errorf("launch ref must be launch.<symbol>.int, got %q", strings.Join(parts, "."))
	}

	if resolver.dispatcher.launchBindings == nil {
		return nil, fmt.Errorf("launch binding %q is not set for this graph.call", parts[0])
	}

	value, ok := resolver.dispatcher.launchBindings[parts[0]]

	if !ok {
		return nil, fmt.Errorf("launch binding %q is not set for this graph.call", parts[0])
	}

	return int(value), nil
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

	property := strings.Join(parts[2:], ".")

	switch property {
	case "shape":
		return resolver.resolveInputDimensions(parts[1], inputTensor)
	case "len":
		dimensions, err := resolver.resolveInputDimensions(parts[1], inputTensor)

		if err != nil {
			return nil, err
		}

		return productInts(dimensions), nil
	default:
		return tensorProperty(inputTensor, property)
	}
}

func (resolver *bindResolver) resolveInputDimensions(
	source string,
	inputTensor tensor.Tensor,
) ([]int, error) {
	inputIndex, err := resolver.inputIndex(source)

	if err != nil {
		return nil, err
	}

	if resolver.dispatcher.workspaces != nil {
		inputName := resolver.node.Inputs[inputIndex]

		if !isBoundaryInput(resolver.dispatcher.graph, inputName) {
			if _, ok := resolver.dispatcher.values.get(inputName); ok {
				return inputTensor.Shape().Dims(), nil
			}
		}

		inputTypes, ok := resolver.dispatcher.workspaces.InputTypesFor(
			resolver.dispatcher.graphName,
			resolver.node.ID,
		)

		if ok && inputIndex < len(inputTypes) && len(inputTypes[inputIndex].ShapeSchema.Dimensions) > 0 {
			shape, err := resolveLiveShape(
				inputTypes[inputIndex].ShapeSchema,
				inputTypes[inputIndex].DType,
				resolver.dispatcher.maxBindings,
				resolver.dispatcher.launchBindings,
			)

			if err != nil {
				return nil, err
			}

			return shape.Dims(), nil
		}
	}

	return substituteLaunchDimensions(
		inputTensor.Shape().Dims(),
		resolver.dispatcher.maxBindings,
		resolver.dispatcher.launchBindings,
	), nil
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
	bias := len(parts) >= 2 && parts[1] == "bias"
	transposed := len(parts) >= 2 && parts[1] == "transposed"
	propertyIndex := 1

	if bias {
		propertyIndex = 2
	}

	if transposed {
		propertyIndex = 2
	}

	if len(parts) <= propertyIndex {
		return nil, fmt.Errorf("weight ref requires a property")
	}

	weightTensor, err := resolver.resolveWeightTensor(transposed, bias)

	if err != nil {
		return nil, err
	}

	return tensorProperty(weightTensor, strings.Join(parts[propertyIndex:], "."))
}

func (resolver *bindResolver) resolveConfigRef(parts []string) (any, error) {
	if len(parts) == 4 && parts[2] == "tensor" {
		tensorName := configString(resolver.node, parts[1], resolver.defaultConfigString(parts[1]))

		if tensorName == "" {
			return nil, fmt.Errorf("config tensor %q is not set", parts[1])
		}

		resident, err := resolver.dispatcher.weights.Lookup(tensorName)

		if err != nil {
			return nil, fmt.Errorf("config tensor %q: %w", tensorName, err)
		}

		return tensorProperty(resident, parts[3])
	}

	if len(parts) != 3 {
		return nil, fmt.Errorf("config ref must be config.<name>.<type>, got %q", strings.Join(parts, "."))
	}

	switch parts[2] {
	case "int":
		return configInt(resolver.node, parts[1], resolver.defaultConfigInt(parts[1])), nil
	case "ints":
		return configInts(resolver.node, parts[1], resolver.defaultConfigInts(parts[1]))
	case "float":
		return float32(configFloat(resolver.node, parts[1], resolver.defaultConfigFloat(parts[1]))), nil
	case "bool":
		return configBool(resolver.node, parts[1], resolver.defaultConfigBool(parts[1])), nil
	case "string":
		return configString(resolver.node, parts[1], resolver.defaultConfigString(parts[1])), nil
	default:
		return nil, fmt.Errorf("unknown config type %q", parts[2])
	}
}

func (resolver *bindResolver) defaultConfigInts(key string) []int {
	value, ok := resolver.bind.ConfigDefaults[key]

	if !ok {
		return nil
	}

	values, err := intSliceDefault(value)

	if err != nil {
		return nil
	}

	return values
}

func intSliceDefault(value any) ([]int, error) {
	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		return intSliceValues(typed)
	default:
		return nil, fmt.Errorf("default config is %T, expected int[]", value)
	}
}

func (resolver *bindResolver) defaultConfigInt(key string) int {
	value, ok := resolver.bind.ConfigDefaults[key]

	if !ok {
		return 0
	}

	asInt, err := scalarInt(value)

	if err != nil {
		return 0
	}

	return asInt
}

func (resolver *bindResolver) defaultConfigFloat(key string) float64 {
	value, ok := resolver.bind.ConfigDefaults[key]

	if !ok {
		return 0
	}

	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return 0
	}
}

func (resolver *bindResolver) defaultConfigBool(key string) bool {
	value, ok := resolver.bind.ConfigDefaults[key]

	if !ok {
		return false
	}

	asBool, ok := value.(bool)

	return ok && asBool
}

func (resolver *bindResolver) defaultConfigString(key string) string {
	value, ok := resolver.bind.ConfigDefaults[key]

	if !ok {
		return ""
	}

	asString, ok := value.(string)

	if !ok {
		return ""
	}

	return asString
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
