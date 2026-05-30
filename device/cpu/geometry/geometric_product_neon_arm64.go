//go:build arm64

package geometry

//go:noescape
func GeometricProductFloat64NEONAsm(left, right, destination *float64)

//go:noescape
func RotorSimilarity128NEONAsm(left, right *float64, count int) float64

func geometricProductFloat64NEON(left, right, destination *float64) {
	GeometricProductFloat64NEONAsm(left, right, destination)
}

func rotorSimilarity128NEON(left, right *float64, count int) float64 {
	return RotorSimilarity128NEONAsm(left, right, count)
}
