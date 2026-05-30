//go:build amd64

package geometry

//go:noescape
func PhaseCouplingFloat16SSE2Asm(
	destination, leftGrowth, rightGrowth *uint16,
	count int,
)

func PhaseCouplingFloat16SSE2(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	if count == 0 {
		return
	}

	PhaseCouplingFloat16SSE2Asm(
		&destination[0], &leftGrowth[0], &rightGrowth[0], count,
	)
}
