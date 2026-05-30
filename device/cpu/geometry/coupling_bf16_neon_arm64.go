//go:build arm64

package geometry

//go:noescape
func PhaseCouplingBFloat16NEONAsm(
	destination, leftGrowth, rightGrowth *uint16,
	count int,
)

func PhaseCouplingBFloat16NEON(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	if count == 0 {
		return
	}

	PhaseCouplingBFloat16NEONAsm(
		&destination[0], &leftGrowth[0], &rightGrowth[0], count,
	)
}
