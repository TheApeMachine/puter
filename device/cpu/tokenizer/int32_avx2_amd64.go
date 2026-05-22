//go:build amd64

package tokenizer

//go:noescape
func TokenizerPackInt32AVX2Asm(dst, src *int32, count int)

func TokenizerPackInt32AVX2(dst, src *int32, count int) {
	if count == 0 {
		return
	}

	TokenizerPackInt32AVX2Asm(dst, src, count)
}
