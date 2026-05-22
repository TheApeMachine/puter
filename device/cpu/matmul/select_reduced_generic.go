//go:build !arm64

package matmul

import "unsafe"

func MatmulBFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	runMatmulReduced(out, left, right, rows, inner, cols, loadBF16, storeBF16)
}

func MatmulFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	runMatmulReduced(out, left, right, rows, inner, cols, loadF16, storeF16)
}
