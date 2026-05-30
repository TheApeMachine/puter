//go:build amd64

package geometry

//go:noescape
func PhaseCouplingFloat16AVX2Asm(
	destination, leftGrowth, rightGrowth *uint16,
	count int,
)

func PhaseCouplingFloat16AVX2(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	if count == 0 {
		return
	}

	PhaseCouplingFloat16AVX2Asm(
		&destination[0], &leftGrowth[0], &rightGrowth[0], count,
	)
}
