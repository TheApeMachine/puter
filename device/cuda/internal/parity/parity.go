package parity

import (
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

/*
Lengths exercises edge alignment, single-vector, and multi-vector paths.
*/
var Lengths = cpuparity.Lengths

/*
AssertFloat32SlicesWithinULP fails when any lane exceeds maxULP from want.
*/
func AssertFloat32SlicesWithinULP(
	testingTB interface {
		Helper()
		Fatal(args ...any)
	},
	got, want []float32,
	maxULP int,
) {
	cpuparity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
}
