//go:build amd64

package losses

//go:noescape
func MseSumFloat32SSE2Asm(predictions, targets *float32, count int) float32

//go:noescape
func MaeSumFloat32SSE2Asm(predictions, targets *float32, count int) float32

func MseSumF32SSE2(predictions, targets *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return MseSumFloat32SSE2Asm(predictions, targets, count)
}

func MaeSumF32SSE2(predictions, targets *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaeSumFloat32SSE2Asm(predictions, targets, count)
}
