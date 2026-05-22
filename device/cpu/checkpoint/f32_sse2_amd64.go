//go:build amd64

package checkpoint

//go:noescape
func CheckpointEncodeFloat32DataSSE2Asm(dst *byte, src *float32, count int)

//go:noescape
func CheckpointDecodeFloat32DataSSE2Asm(dst *float32, src *byte, count int)

func checkpointEncodeFloat32DataSSE2(dst []byte, src []float32) {
	CheckpointEncodeFloat32DataSSE2Asm(&dst[0], &src[0], len(src))
}

func checkpointDecodeFloat32DataSSE2(dst []float32, src []byte) {
	CheckpointDecodeFloat32DataSSE2Asm(&dst[0], &src[0], len(dst))
}
