package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func isIntrinsicMethod(method string) bool {
	switch method {
	case "shape.view_as_heads", "shape.merge_heads", "shape.last_token":
		return true
	default:
		return false
	}
}

func runIntrinsic(resolver *bindResolver) (any, error) {
	switch resolver.bind.Method {
	case "shape.view_as_heads", "shape.merge_heads":
		return runReshapeIntrinsic(resolver)
	case "shape.last_token":
		return runLastTokenIntrinsic(resolver)
	default:
		return nil, fmt.Errorf("unknown intrinsic %q", resolver.bind.Method)
	}
}

func runReshapeIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	if input.Len() != resolver.outputShape.Len() {
		return nil, fmt.Errorf(
			"reshape element count mismatch: input %d, output %d",
			input.Len(), resolver.outputShape.Len(),
		)
	}

	return input.Reshape(resolver.outputShape.Dims())
}

func runLastTokenIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	dimensions := substituteLaunchDimensions(
		input.Shape().Dims(),
		resolver.dispatcher.maxBindings,
		resolver.dispatcher.launchBindings,
	)

	if len(dimensions) < 2 {
		return nil, fmt.Errorf("shape.last_token input must have rank >= 2, got %d", len(dimensions))
	}

	rows := dimensions[0]

	if rows < 1 {
		return nil, fmt.Errorf("shape.last_token input has zero rows")
	}

	rowElements := productInts(dimensions[1:])
	start := (rows - 1) * rowElements

	slice, err := input.Slice(start, rowElements)

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if err := copyTensorStorage(output, slice, rowElements); err != nil {
		return nil, err
	}

	return output, nil
}

func copyTensorStorage(destination tensor.Tensor, source tensor.Tensor, elements int) error {
	if destination.DType() != source.DType() {
		return fmt.Errorf(
			"copy tensor storage dtype mismatch: destination %s, source %s",
			destination.DType(),
			source.DType(),
		)
	}

	elementSize, err := source.DType().Size()

	if err != nil {
		return err
	}

	byteCount := elements * elementSize
	destinationPointer, _, err := pointerOf(destination)

	if err != nil {
		return err
	}

	sourcePointer, _, err := pointerOf(source)

	if err != nil {
		return err
	}

	copy(
		unsafe.Slice((*byte)(destinationPointer), byteCount),
		unsafe.Slice((*byte)(sourcePointer), byteCount),
	)

	return nil
}
