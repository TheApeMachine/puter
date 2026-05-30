package geometry

var (
	geometricProductKernel = func() func(left, right, destination *float64) {
		return pickGeometricProductKernel(geometricProductFuncs)
	}()

	rotorSimilarityKernel = func() func(left, right *float64, count int) float64 {
		return pickRotorSimilarityKernel(rotorSimilarityFuncs)
	}()
)

func geometricProductMultivector(left, right Multivector) Multivector {
	var result Multivector

	geometricProductKernel(&left[0], &right[0], &result[0])

	return result
}

func rotorSimilarityAverage(leftRotor, rightRotor PhaseRotor) float64 {
	if len(leftRotor) == 0 {
		return 0
	}

	return rotorSimilarityKernel(&leftRotor[0][0], &rightRotor[0][0], len(leftRotor))
}
