//go:build arm64

package tokenizer

//go:noescape
func TokenizerPackInt32NEONAsm(dst, src *int32, count int)

/*
TokenizerPackInt32NEON copies count int32 lanes from src to dst.
*/
func TokenizerPackInt32NEON(dst, src *int32, count int) {
	if count == 0 {
		return
	}

	TokenizerPackInt32NEONAsm(dst, src, count)
}
