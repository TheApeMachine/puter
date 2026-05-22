//go:build amd64

package attention

//go:noescape
func FlashAttentionOnlineUpdateAVX2Asm(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
)

//go:noescape
func FlashAttentionScaleAVX2Asm(
	out, acc *float32,
	invNormalizer float32,
	length int,
)

func flashAttentionOnlineUpdateAVX2(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionOnlineUpdateAVX2Asm(acc, valueRow, alpha, shifted, length)
}

func flashAttentionScaleAVX2(
	out, acc *float32,
	invNormalizer float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionScaleAVX2Asm(out, acc, invNormalizer, length)
}
