//go:build arm64

package geometry

func neonPhaseCouplingAvailable() bool {
	for _, candidate := range phaseCouplingFloat32Funcs {
		if candidate.name == "neon" && candidate.available {
			return true
		}
	}

	return false
}
