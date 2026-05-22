//go:build amd64

package reduction

//go:noescape
func SumFloat32AVX2Asm(values *float32, count int) float32

//go:noescape
func ProdFloat32AVX2Asm(values *float32, count int) float32

//go:noescape
func ReduceMaxFloat32AVX2Asm(values *float32, count int) float32

//go:noescape
func ReduceMinFloat32AVX2Asm(values *float32, count int) float32

//go:noescape
func L1NormFloat32AVX2Asm(values *float32, count int) float32

func SumF32AVX2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return SumFloat32AVX2Asm(values, count)
}

func ProdF32AVX2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat32AVX2Asm(values, count)
}

func MaxF32AVX2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMaxFloat32AVX2Asm(values, count)
}

func MinF32AVX2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMinFloat32AVX2Asm(values, count)
}

func L1NormF32AVX2(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat32AVX2Asm(values, count)
}
