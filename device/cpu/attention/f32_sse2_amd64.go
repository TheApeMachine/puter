//go:build amd64

package attention

//go:noescape
func FlashAttentionOnlineUpdateSSE2Asm(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
)

//go:noescape
func FlashAttentionScaleSSE2Asm(
	out, acc *float32,
	invNormalizer float32,
	length int,
)

func flashAttentionOnlineUpdateSSE2(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionOnlineUpdateSSE2Asm(acc, valueRow, alpha, shifted, length)
}

func flashAttentionScaleSSE2(
	out, acc *float32,
	invNormalizer float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionScaleSSE2Asm(out, acc, invNormalizer, length)
}
