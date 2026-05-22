//go:build arm64

package checkpoint

//go:noescape
func CheckpointEncodeFloat32DataNEONAsm(dst *byte, src *float32, count int)

//go:noescape
func CheckpointDecodeFloat32DataNEONAsm(dst *float32, src *byte, count int)

/*
CheckpointEncodeFloat32DataNEON writes count float32 lanes from src into dst as little-endian bytes.
*/
func CheckpointEncodeFloat32DataNEON(dst *byte, src *float32, count int) {
	if count == 0 {
		return
	}

	CheckpointEncodeFloat32DataNEONAsm(dst, src, count)
}

/*
CheckpointDecodeFloat32DataNEON reads count float32 lanes from little-endian bytes in src into dst.
*/
func CheckpointDecodeFloat32DataNEON(dst *float32, src *byte, count int) {
	if count == 0 {
		return
	}

	CheckpointDecodeFloat32DataNEONAsm(dst, src, count)
}
