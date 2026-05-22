//go:build !darwin || !cgo

package metal

import "math"

/*
normMetalFMAFloat32 is a best-effort fma when libm fmaf is unavailable.
*/
func normMetalFMAFloat32(a float32, b float32, c float32) float32 {
	return float32(math.FMA(float64(a), float64(b), float64(c)))
}
