//go:build arm64

package active_inference

import "math"

/*
activeInferenceStdLogF64 is invoked from typed FE/EFE assembly.
It matches FreeEnergy*Scalar / ExpectedFreeEnergy*Scalar log semantics.
*/
func activeInferenceStdLogF64(value float64) float64 {
	if value <= 0 {
		return math.NaN()
	}

	return math.Log(value)
}
