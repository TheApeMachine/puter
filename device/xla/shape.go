package xla

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
ValidateShape checks that a tensor shape is dense-rank compatible with XLA lowering.
*/
func ValidateShape(shape tensor.Shape) error {
	if shape.Rank() == 0 && shape.Len() != 1 {
		return &loweringError{message: "invalid scalar XLA shape"}
	}

	for _, dimension := range shape.Dims() {
		if dimension < 0 {
			return &loweringError{message: "negative XLA shape dimension"}
		}
	}

	return nil
}

/*
ElementCount returns the dense element count for a validated shape.
*/
func ElementCount(shape tensor.Shape) (int64, error) {
	if err := ValidateShape(shape); err != nil {
		return 0, err
	}

	return int64(shape.Len()), nil
}

/*
BroadcastShape returns the broadcast result of left and right dense shapes.
Trailing dimensions are aligned from the right per NumPy rules.
*/
func BroadcastShape(left tensor.Shape, right tensor.Shape) (tensor.Shape, error) {
	leftDimensions := left.Dims()
	rightDimensions := right.Dims()
	leftRank := len(leftDimensions)
	rightRank := len(rightDimensions)
	outputRank := leftRank

	if rightRank > outputRank {
		outputRank = rightRank
	}

	outputDimensions := make([]int, outputRank)

	for offset := 0; offset < outputRank; offset++ {
		leftDimension := 1
		rightDimension := 1

		leftIndex := leftRank - 1 - offset
		rightIndex := rightRank - 1 - offset

		if leftIndex >= 0 {
			leftDimension = leftDimensions[leftIndex]
		}

		if rightIndex >= 0 {
			rightDimension = rightDimensions[rightIndex]
		}

		if leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1 {
			return tensor.Shape{}, &loweringError{message: "incompatible XLA broadcast shapes"}
		}

		if leftDimension > rightDimension {
			outputDimensions[outputRank-1-offset] = leftDimension
			continue
		}

		outputDimensions[outputRank-1-offset] = rightDimension
	}

	return tensor.NewShape(outputDimensions)
}
