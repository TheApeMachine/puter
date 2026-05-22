//go:build arm64

package matmul

import (
	"unsafe"
)

//go:noescape
func MatmulRowBF16NEONAsm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowFP16NEONAsm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

func MatmulBFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	colsBlock := cols &^ 3

	if colsBlock > 0 {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			rowOffset := rowIndex * cols
			innerOffset := rowIndex * inner

			MatmulRowBF16NEONAsm(
				(*uint16)(unsafe.Add(out, uintptr(rowOffset)*2)),
				(*uint16)(unsafe.Add(left, uintptr(innerOffset)*2)),
				(*uint16)(right),
				inner,
				colsBlock,
				cols,
			)
		}
	}

	if colsBlock == cols {
		return
	}

	runMatmulReducedCols(
		out, left, right,
		rows, inner, cols,
		loadBF16, storeBF16,
		colsBlock,
	)
}

func MatmulFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	colsBlock := cols &^ 3

	if colsBlock > 0 {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			rowOffset := rowIndex * cols
			innerOffset := rowIndex * inner

			MatmulRowFP16NEONAsm(
				(*uint16)(unsafe.Add(out, uintptr(rowOffset)*2)),
				(*uint16)(unsafe.Add(left, uintptr(innerOffset)*2)),
				(*uint16)(right),
				inner,
				colsBlock,
				cols,
			)
		}
	}

	if colsBlock == cols {
		return
	}

	runMatmulReducedCols(
		out, left, right,
		rows, inner, cols,
		loadF16, storeF16,
		colsBlock,
	)
}
