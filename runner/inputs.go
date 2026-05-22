package runner

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/weights"
)

func bindProgramInputs(
	memory tensor.Backend,
	manifestGraph *ast.Graph,
	computeGraph *ir.Graph,
	programInputs map[string]any,
	tensorWorkspace *workspace,
) error {
	inputSet := make(map[string]struct{}, len(manifestGraph.Inputs))

	for _, inputName := range manifestGraph.Inputs {
		inputSet[inputName] = struct{}{}
	}

	for _, node := range computeGraph.Nodes() {
		if node.OpType() != ir.OpInput {
			continue
		}

		if _, ok := inputSet[node.ID()]; !ok {
			continue
		}

		value, ok := programInputs[node.ID()]

		if !ok {
			return fmt.Errorf("runner: missing graph input %q", node.ID())
		}

		resident, err := residentInputTensor(memory, node, value)

		if err != nil {
			return fmt.Errorf("runner: bind input %q: %w", node.ID(), err)
		}

		tensorWorkspace.Store(node.ID(), resident)
	}

	return nil
}

func residentInputTensor(
	memory tensor.Backend,
	node *ir.Node,
	value any,
) (tensor.Tensor, error) {
	if resident, ok := value.(tensor.Tensor); ok {
		return resident, nil
	}

	storageDType := node.ValueType().DType

	if storageDType == dtype.Invalid {
		storageDType = dtype.Float32
	}

	switch typed := value.(type) {
	case []int:
		return uploadInt32Indices(memory, typed)
	case []int32:
		return uploadInt32Slice(memory, typed)
	case []float32:
		return uploadFloat32Slice(memory, node.Shape(), typed)
	case []float64:
		elements := make([]float32, len(typed))

		for index, element := range typed {
			elements[index] = float32(element)
		}

		return uploadFloat32Slice(memory, node.Shape(), elements)
	default:
		return nil, fmt.Errorf("unsupported input type %T", value)
	}
}

func uploadInt32Indices(memory tensor.Backend, values []int) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], uint32(value))
	}

	return memory.Upload(shape, dtype.Int32, buffer)
}

func uploadInt32Slice(memory tensor.Backend, values []int32) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], uint32(value))
	}

	return memory.Upload(shape, dtype.Int32, buffer)
}

func uploadFloat32Slice(
	memory tensor.Backend,
	shape tensor.Shape,
	values []float32,
) (tensor.Tensor, error) {
	if !shape.Valid() {
		var err error

		shape, err = tensor.NewShape([]int{len(values)})

		if err != nil {
			return nil, err
		}
	}

	buffer := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
	}

	return memory.Upload(shape, dtype.Float32, buffer)
}

func collectProgramOutputs(
	memory tensor.Backend,
	manifestGraph *ast.Graph,
	tensorWorkspace *workspace,
) (map[string]any, error) {
	outputs := make(map[string]any, len(manifestGraph.Outputs))

	for outputName, nodeID := range manifestGraph.Outputs {
		value, ok := tensorWorkspace.Load(nodeID)

		if !ok {
			return nil, fmt.Errorf("runner: missing output tensor for node %q", nodeID)
		}

		hostValues, err := downloadFloat32Vector(memory, value)

		if err != nil {
			return nil, fmt.Errorf("runner: output %q: %w", outputName, err)
		}

		outputs[outputName] = hostValues
	}

	return outputs, nil
}

func downloadFloat32Vector(memory tensor.Backend, value tensor.Tensor) ([]float32, error) {
	storageDType, raw, err := memory.Download(value)

	if err != nil {
		return nil, err
	}

	if storageDType == dtype.Float32 {
		elementCount := len(raw) / 4
		elements := make([]float32, elementCount)

		for index := range elementCount {
			elements[index] = math.Float32frombits(
				binary.LittleEndian.Uint32(raw[index*4 : index*4+4]),
			)
		}

		return elements, nil
	}

	if storageDType.IsFloat() {
		return convert.BytesToFloat32(storageDType, raw)
	}

	return nil, fmt.Errorf("expected float output, got %s", storageDType)
}

func weightTensorName(node *ir.Node) string {
	metadata := node.Metadata()

	weightName, ok := metadata["weight_name"].(string)

	if !ok || weightName == "" {
		return ""
	}

	return weightName
}

func weightsPath(manifestGraph *ast.Graph) string {
	if manifestGraph == nil || manifestGraph.Metadata == nil {
		return ""
	}

	weightsPathValue, ok := manifestGraph.Metadata["weights_path"].(string)

	if !ok {
		return ""
	}

	return weightsPathValue
}

func readWeightBytes(path string, tensorName string) ([]byte, weights.TensorMeta, error) {
	return weights.ReadTensor(path, tensorName)
}
