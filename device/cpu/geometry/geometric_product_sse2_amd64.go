//go:build amd64

package geometry

//go:noescape
func GeometricProductFloat64SSE2Asm(left, right, destination *float64)

//go:noescape
func RotorSimilarity128SSE2Asm(left, right *float64, count int) float64

func geometricProductFloat64SSE2(left, right, destination *float64) {
	GeometricProductFloat64SSE2Asm(left, right, destination)
}

func rotorSimilarity128SSE2(left, right *float64, count int) float64 {
	return RotorSimilarity128SSE2Asm(left, right, count)
}
