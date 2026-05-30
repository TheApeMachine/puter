package geometry

import (
	"github.com/theapemachine/manifesto/dtype"
)

/*
PhaseCouplingBFloat16Scalar writes directional coupling between surprisal
velocities into destination[i] for each index.
*/
func PhaseCouplingBFloat16Scalar(
	destination, leftGrowth, rightGrowth []uint16,
) {
	for index := range destination {
		leftValue := dtype.BF16(leftGrowth[index])
		rightValue := dtype.BF16(rightGrowth[index])
		destination[index] = uint16(
			scalarPhaseCouplingReferenceBF16(leftValue, rightValue),
		)
	}
}
