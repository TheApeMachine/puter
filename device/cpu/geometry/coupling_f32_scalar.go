package geometry

import "math"

const phaseCouplingMagEps = float32(0.01)

/*
PhaseCouplingFloat32Scalar writes directional coupling between surprisal
velocities into destination[i] for each index.
*/
func PhaseCouplingFloat32Scalar(
	destination, leftGrowth, rightGrowth []float32,
) {
	for index := range destination {
		leftValue := leftGrowth[index]
		rightValue := rightGrowth[index]

		absLeft := leftValue

		if absLeft < 0 {
			absLeft = -absLeft
		}

		absRight := rightValue

		if absRight < 0 {
			absRight = -absRight
		}

		geometricMean := float32(math.Sqrt(float64(absLeft * absRight)))

		if geometricMean < phaseCouplingMagEps {
			destination[index] = 0
			continue
		}

		destination[index] = (leftValue * rightValue) / (geometricMean * geometricMean)
	}
}

func scalarPhaseCouplingReference(
	leftValue, rightValue float32,
) float32 {
	absLeft := leftValue

	if absLeft < 0 {
		absLeft = -absLeft
	}

	absRight := rightValue

	if absRight < 0 {
		absRight = -absRight
	}

	geometricMean := float32(math.Sqrt(float64(absLeft * absRight)))

	if geometricMean < phaseCouplingMagEps {
		return 0
	}

	return (leftValue * rightValue) / (geometricMean * geometricMean)
}
