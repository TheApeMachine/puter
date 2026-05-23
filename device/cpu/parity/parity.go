package parity

import (
	"math"
)

/*
Lengths exercises edge alignment, single-vector, and multi-vector paths.
*/
var Lengths = []int{1, 7, 64, 1024, 8192}

/*
FatalTesting is implemented by *testing.T and *testing.B.
*/
type FatalTesting interface {
	Helper()
	Fatal(args ...any)
}

/*
Float32ULPDistance returns the ULP gap between two float32 values.
*/
func Float32ULPDistance(left, right float32) int {
	leftBits := float32BitsOrdered(left)
	rightBits := float32BitsOrdered(right)

	if leftBits > rightBits {
		leftBits, rightBits = rightBits, leftBits
	}

	return int(rightBits - leftBits)
}

func float32BitsOrdered(value float32) uint32 {
	bits := math.Float32bits(value)

	const signBit = uint32(1) << 31

	if bits&signBit != 0 {
		return signBit - bits
	}

	return bits
}

const nearZeroFloat32 = 1e-11

/*
AssertFloat32SlicesWithinULP fails when any lane exceeds maxULP from want.
*/
func AssertFloat32SlicesWithinULP(
	testingTB FatalTesting,
	got, want []float32,
	maxULP int,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatal("length mismatch got=", len(got), " want=", len(want))
	}

	for index := range got {
		if float32LanesMatch(got[index], want[index], maxULP) {
			continue
		}

		testingTB.Fatal(
			"lane ", index,
			" got=", got[index],
			" want=", want[index],
			" ulp=", Float32ULPDistance(got[index], want[index]),
			" max=", maxULP,
		)
	}
}

func float32LanesMatch(left, right float32, maxULP int) bool {
	if left == right {
		return true
	}

	if math.IsNaN(float64(left)) && math.IsNaN(float64(right)) {
		return true
	}

	leftAbs := math.Abs(float64(left))
	rightAbs := math.Abs(float64(right))

	if leftAbs < nearZeroFloat32 && rightAbs < nearZeroFloat32 {
		return true
	}

	return Float32ULPDistance(left, right) <= maxULP
}
