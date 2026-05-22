//go:build darwin && cgo

package metal

/*
#include <math.h>
*/
import "C"

import "math"

/*
hawkesExpMetalReference32 matches metal_hawkes_exp32 in hawkes_markov.metal.
*/
func hawkesExpMetalReference32(value float32) float32 {
	const log2e float32 = 1.4426950408889634
	const ln2 float32 = 0.6931471805599453

	scaled := value * log2e
	roundedK := float32(C.rintf(C.float(scaled)))
	fraction := value - roundedK*ln2
	poly := float32(0.00019841270)

	poly = normMetalFMAFloat32(fraction, poly, 0.0013888889)
	poly = normMetalFMAFloat32(fraction, poly, 0.008333334)
	poly = normMetalFMAFloat32(fraction, poly, 0.041666667)
	poly = normMetalFMAFloat32(fraction, poly, 0.16666667)
	poly = normMetalFMAFloat32(fraction, poly, 0.5)
	poly = normMetalFMAFloat32(fraction, poly, 1.0)
	poly = normMetalFMAFloat32(fraction, poly, 1.0)

	exponentInt := int32(roundedK)
	scaleBits := uint32(exponentInt+127) << 23

	return poly * math.Float32frombits(scaleBits)
}
