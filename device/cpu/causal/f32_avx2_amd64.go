//go:build amd64

package causal

//go:noescape
func CateFloat32AVX2Asm(treated, control, out *float32, count int)

//go:noescape
func CounterfactualFloat32AVX2Asm(
	out, observedY, observedX, counterfactualX *float32,
	slope float32,
	count int,
)

//go:noescape
func StridedDotFloat32AVX2Asm(values *float32, stride int, weights *float32, count int) float32

func cateF32AVX2(treated, control, out []float32) {
	if len(out) == 0 {
		return
	}

	CateFloat32AVX2Asm(&treated[0], &control[0], &out[0], len(out))
}

func counterfactualF32AVX2(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	elementCount := len(out)

	if elementCount == 0 {
		return
	}

	CounterfactualFloat32AVX2Asm(
		&out[0], &observedY[0], &observedX[0], &counterfactualX[0],
		slope, elementCount,
	)
}

func stridedDotF32AVX2(values []float32, stride int, weights []float32, elementCount int) float32 {
	if elementCount == 0 {
		return 0
	}

	return StridedDotFloat32AVX2Asm(&values[0], stride, &weights[0], elementCount)
}
