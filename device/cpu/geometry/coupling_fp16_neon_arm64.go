//go:build arm64

package geometry

//go:noescape
func PhaseCouplingFloat16NEONAsm(
	destination, leftGrowth, rightGrowth *uint16,
	count int,
)

func PhaseCouplingFloat16NEON(
	destination, leftGrowth, rightGrowth []uint16,
	count int,
) {
	if count == 0 {
		return
	}

	vectorCount := count &^ 7

	if vectorCount > 0 {
		PhaseCouplingFloat16NEONAsm(
			&destination[0], &leftGrowth[0], &rightGrowth[0], vectorCount,
		)
	}

	tailCount := count - vectorCount

	if tailCount == 0 {
		return
	}

	PhaseCouplingFloat16Scalar(
		destination[vectorCount:count],
		leftGrowth[vectorCount:count],
		rightGrowth[vectorCount:count],
	)
}
