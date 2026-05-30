//go:build amd64

package geometry

//go:noescape
func PhaseCouplingFloat32AVX512Asm(
	destination, leftGrowth, rightGrowth *float32,
	count int,
)

func PhaseCouplingFloat32AVX512(
	destination, leftGrowth, rightGrowth []float32,
	count int,
) {
	if count == 0 {
		return
	}

	PhaseCouplingFloat32AVX512Asm(
		&destination[0], &leftGrowth[0], &rightGrowth[0], count,
	)
}
