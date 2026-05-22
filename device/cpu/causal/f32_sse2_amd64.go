//go:build amd64

package causal

//go:noescape
func CateFloat32SSE2Asm(treated, control, out *float32, count int)

//go:noescape
func CounterfactualFloat32SSE2Asm(
	out, observedY, observedX, counterfactualX *float32,
	slope float32,
	count int,
)

//go:noescape
func StridedDotFloat32SSE2Asm(values *float32, stride int, weights *float32, count int) float32

func cateF32SSE2(treated, control, out []float32) {
	if len(out) == 0 {
		return
	}

	CateFloat32SSE2Asm(&treated[0], &control[0], &out[0], len(out))
}

func counterfactualF32SSE2(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	elementCount := len(out)

	if elementCount == 0 {
		return
	}

	CounterfactualFloat32SSE2Asm(
		&out[0], &observedY[0], &observedX[0], &counterfactualX[0],
		slope, elementCount,
	)
}

func stridedDotF32SSE2(values []float32, stride int, weights []float32, elementCount int) float32 {
	if elementCount == 0 {
		return 0
	}

	return StridedDotFloat32SSE2Asm(&values[0], stride, &weights[0], elementCount)
}
