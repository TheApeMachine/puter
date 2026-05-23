//go:build xla

package parity

import (
	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

/*
Lengths exercises edge alignment, single-vector, and multi-vector paths.
*/
var Lengths = cpuparity.Lengths

/*
FloatParityDTypes lists native float dtypes exercised by XLA parity tests.
*/
var FloatParityDTypes = []dtype.DType{
	dtype.Float64,
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
	dtype.Float8E4M3,
	dtype.Float8E5M2,
}

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
