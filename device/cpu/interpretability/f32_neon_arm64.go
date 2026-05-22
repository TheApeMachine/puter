//go:build arm64

package interpretability

//go:noescape
func ActivationSteerFloat32NEONAsm(
	destination, base, direction *float32,
	coefficient float32,
	count int,
)

/*
ActivationSteerFloat32NEON computes dst[i] = base[i] + coefficient * direction[i].
*/
func ActivationSteerFloat32NEON(
	destination, base, direction *float32,
	coefficient float32,
	count int,
) {
	if count == 0 {
		return
	}

	ActivationSteerFloat32NEONAsm(destination, base, direction, coefficient, count)
}
