package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

type tensorBytesWriter interface {
	WriteTensorBytes(target tensor.Tensor, bytesIn []byte) error
}

func seedGraphInputs(
	backend *Backend,
	graphName string,
	inputs map[string]any,
	memory tensor.Backend,
) (map[string]any, error) {
	if backend == nil || backend.workspaces == nil || len(inputs) == 0 {
		return inputs, nil
	}

	seeded := make(map[string]any, len(inputs))

	for name, value := range inputs {
		boundary, ok := backend.workspaces.BoundaryInput(graphName, name)

		if !ok {
			seeded[name] = value
			continue
		}

		source, ok := value.(tensor.Tensor)

		if !ok {
			seeded[name] = value
			continue
		}

		if err := materializeGraphInput(memory, boundary, source); err != nil {
			return nil, fmt.Errorf("graph input %q: %w", name, err)
		}

		seeded[name] = boundary
	}

	return seeded, nil
}

func materializeGraphInput(
	memory tensor.Backend,
	target tensor.Tensor,
	source tensor.Tensor,
) error {
	if target == nil || source == nil {
		return fmt.Errorf("target and source tensors are required")
	}

	if target.Len() != source.Len() {
		return fmt.Errorf(
			"target length %d does not match source length %d",
			target.Len(),
			source.Len(),
		)
	}

	if target.DType() == source.DType() {
		return copyMatchingTensor(memory, target, source)
	}

	sourceDType, sourceBytes, err := memory.Download(source)

	if err != nil {
		return fmt.Errorf("read source: %w", err)
	}

	float64Values, err := convert.BytesToFloat64(sourceDType, sourceBytes)

	if err != nil {
		return fmt.Errorf("decode source: %w", err)
	}

	targetBytes, err := encodeTensorBytes(target.DType(), float64Values)

	if err != nil {
		return err
	}

	writer, ok := memory.(tensorBytesWriter)

	if !ok {
		return fmt.Errorf("memory backend cannot write resident tensor bytes")
	}

	return writer.WriteTensorBytes(target, targetBytes)
}

func copyMatchingTensor(
	memory tensor.Backend,
	target tensor.Tensor,
	source tensor.Tensor,
) error {
	sourceDType, sourceBytes, err := memory.Download(source)

	if err != nil {
		return fmt.Errorf("read source: %w", err)
	}

	writer, ok := memory.(tensorBytesWriter)

	if !ok {
		return fmt.Errorf("memory backend cannot write resident tensor bytes")
	}

	if sourceDType != target.DType() {
		return fmt.Errorf(
			"copy dtype mismatch: target %s, source %s",
			target.DType(),
			sourceDType,
		)
	}

	return writer.WriteTensorBytes(target, sourceBytes)
}

func encodeTensorBytes(targetDType dtype.DType, values []float64) ([]byte, error) {
	switch targetDType {
	case dtype.Float64:
		return convert.Float64ToBytes(values), nil
	case dtype.Float32:
		out := make([]float32, len(values))

		for index, value := range values {
			out[index] = float32(value)
		}

		return convert.Float32ToBytes(out), nil
	case dtype.BFloat16:
		out := make([]dtype.BF16, len(values))

		for index, value := range values {
			out[index] = dtype.NewBfloat16FromFloat32(float32(value))
		}

		return convert.BFloat16ToBytes(out), nil
	default:
		return nil, fmt.Errorf("unsupported target dtype %s", targetDType)
	}
}
