package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

func runReshapeIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	liveInput, err := resolver.liveInputTensor("0", input)

	if err != nil {
		return nil, err
	}

	if liveInput.Len() != resolver.outputShape.Len() {
		return nil, fmt.Errorf(
			"reshape element count mismatch: input %d, output %d",
			liveInput.Len(), resolver.outputShape.Len(),
		)
	}

	return liveInput.Reshape(resolver.outputShape.Dims())
}

func (resolver *bindResolver) liveInputTensor(source string, input tensor.Tensor) (tensor.Tensor, error) {
	physicalDimensions := input.Shape().Dims()
	liveDimensions, err := resolver.resolveInputDimensions(source, input)

	if err != nil {
		return nil, err
	}

	if dimensionsEqual(physicalDimensions, liveDimensions) {
		return input, nil
	}

	liveLength := productInts(liveDimensions)

	if liveLength > input.Len() {
		return nil, fmt.Errorf(
			"live shape %v has %d elements, exceeds planned shape %v with %d elements",
			liveDimensions, liveLength, physicalDimensions, input.Len(),
		)
	}

	if liveLength == input.Len() {
		return input.Reshape(liveDimensions)
	}

	if !liveShapeIsContiguousPrefix(physicalDimensions, liveDimensions) {
		return nil, fmt.Errorf(
			"live shape %v is not a contiguous prefix of planned shape %v",
			liveDimensions, physicalDimensions,
		)
	}

	view, err := input.Slice(0, liveLength)

	if err != nil {
		return nil, fmt.Errorf(
			"slice planned shape %v to live shape %v: %w",
			physicalDimensions, liveDimensions, err,
		)
	}

	return view.Reshape(liveDimensions)
}

func dimensionsEqual(left []int, right []int) bool {
	if len(left) != len(right) {
		return false
	}

	for index, leftValue := range left {
		if leftValue != right[index] {
			return false
		}
	}

	return true
}

func liveShapeIsContiguousPrefix(physicalDimensions []int, liveDimensions []int) bool {
	if len(physicalDimensions) != len(liveDimensions) {
		return false
	}

	leadingLiveProduct := 1

	for index, physicalDimension := range physicalDimensions {
		liveDimension := liveDimensions[index]

		if liveDimension > physicalDimension {
			return false
		}

		if liveDimension < physicalDimension && leadingLiveProduct != 1 {
			return false
		}

		leadingLiveProduct *= liveDimension
	}

	return true
}
