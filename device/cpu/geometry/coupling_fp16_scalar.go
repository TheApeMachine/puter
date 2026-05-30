package geometry

import (
	"github.com/theapemachine/manifesto/dtype"
)

/*
PhaseCouplingFloat16Scalar writes directional coupling between surprisal
velocities into destination[i] for each index.
*/
func PhaseCouplingFloat16Scalar(
	destination, leftGrowth, rightGrowth []uint16,
) {
	for index := range destination {
		leftValue := dtype.F16(leftGrowth[index])
		rightValue := dtype.F16(rightGrowth[index])
		destination[index] = uint16(
			scalarPhaseCouplingReferenceF16(leftValue, rightValue),
		)
	}
}
