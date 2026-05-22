//go:build amd64

package checkpoint

//go:noescape
func CheckpointEncodeFloat32DataAVX2Asm(dst *byte, src *float32, count int)

//go:noescape
func CheckpointDecodeFloat32DataAVX2Asm(dst *float32, src *byte, count int)

func checkpointEncodeFloat32DataAVX2(dst []byte, src []float32) {
	CheckpointEncodeFloat32DataAVX2Asm(&dst[0], &src[0], len(src))
}

func checkpointDecodeFloat32DataAVX2(dst []float32, src []byte) {
	CheckpointDecodeFloat32DataAVX2Asm(&dst[0], &src[0], len(dst))
}
