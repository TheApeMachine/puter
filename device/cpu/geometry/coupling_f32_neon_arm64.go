//go:build arm64

package geometry

//go:noescape
func PhaseCouplingFloat32NEONAsm(
	destination, leftGrowth, rightGrowth *float32,
	count int,
)

func PhaseCouplingFloat32NEON(
	destination, leftGrowth, rightGrowth []float32,
	count int,
) {
	if count == 0 {
		return
	}

	vectorCount := count &^ 3

	if vectorCount > 0 {
		PhaseCouplingFloat32NEONAsm(
			&destination[0], &leftGrowth[0], &rightGrowth[0], vectorCount,
		)
	}

	tailCount := count - vectorCount

	if tailCount == 0 {
		return
	}

	PhaseCouplingFloat32Scalar(
		destination[vectorCount:count],
		leftGrowth[vectorCount:count],
		rightGrowth[vectorCount:count],
	)
}
