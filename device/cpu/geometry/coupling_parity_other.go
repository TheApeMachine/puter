//go:build !arm64

package geometry

func neonPhaseCouplingAvailable() bool {
	return false
}
