//go:build amd64

package reduction

//go:noescape
func SumFloat32SSE2Asm(values *float32, count int) float32

//go:noescape
func ProdFloat32SSE2Asm(values *float32, count int) float32

//go:noescape
func ReduceMaxFloat32SSE2Asm(values *float32, count int) float32

//go:noescape
func ReduceMinFloat32SSE2Asm(values *float32, count int) float32

//go:noescape
func L1NormFloat32SSE2Asm(values *float32, count int) float32

func SumF32SSE2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return SumFloat32SSE2Asm(values, count)
}

func ProdF32SSE2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat32SSE2Asm(values, count)
}

func MaxF32SSE2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMaxFloat32SSE2Asm(values, count)
}

func MinF32SSE2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMinFloat32SSE2Asm(values, count)
}

func L1NormF32SSE2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat32SSE2Asm(values, count)
}
