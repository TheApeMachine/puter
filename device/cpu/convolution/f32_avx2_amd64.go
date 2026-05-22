//go:build amd64

package convolution

//go:noescape
func ConvPatchDotFloat32AVX2Asm(weight, patch *float32, length int) float32

func ConvPatchDotF32AVX2(weight, patch *float32, length int) float32 {
	if length == 0 {
		return 0
	}

	return ConvPatchDotFloat32AVX2Asm(weight, patch, length)
}
