//go:build amd64

package interpretability

//go:noescape
func ActivationSteerFloat32AVX2Asm(
	destination, base, direction *float32,
	coefficient float32,
	count int,
)

func ActivationSteerFloat32AVX2(
	destination, base, direction *float32,
	coefficient float32,
	count int,
) {
	if count == 0 {
		return
	}

	ActivationSteerFloat32AVX2Asm(destination, base, direction, coefficient, count)
}

func activationSteerFloat32AVX2(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32AVX2(
		&destination[0], &base[0], &direction[0], coefficient, count,
	)
}
