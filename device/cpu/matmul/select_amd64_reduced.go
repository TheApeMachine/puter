//go:build amd64

package matmul

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func MatmulRowBF16AVX512Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowBF16AVX2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowBF16SSE2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowFP16AVX512Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowFP16AVX2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

//go:noescape
func MatmulRowFP16SSE2Asm(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

func MatmulBFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	if cpu.X86.HasAVX512F {
		dispatchReducedMatmulRows(
			MatmulRowBF16AVX512Asm, out, left, right,
			rows, inner, cols, 8,
			loadBF16, storeBF16,
		)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		dispatchReducedMatmulRows(
			MatmulRowBF16AVX2Asm, out, left, right,
			rows, inner, cols, 8,
			loadBF16, storeBF16,
		)

		return
	}

	if cpu.X86.HasSSE2 {
		dispatchReducedMatmulRows(
			MatmulRowBF16SSE2Asm, out, left, right,
			rows, inner, cols, 4,
			loadBF16, storeBF16,
		)

		return
	}

	runMatmulReduced(out, left, right, rows, inner, cols, loadBF16, storeBF16)
}

func MatmulFloat16Native(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
) {
	if !(cpu.X86.HasAVX2 || cpu.X86.HasAVX512F) {
		runMatmulReduced(out, left, right, rows, inner, cols, loadF16, storeF16)
		return
	}

	if cpu.X86.HasAVX512F {
		dispatchReducedMatmulRows(
			MatmulRowFP16AVX512Asm, out, left, right,
			rows, inner, cols, 8,
			loadF16, storeF16,
		)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		dispatchReducedMatmulRows(
			MatmulRowFP16AVX2Asm, out, left, right,
			rows, inner, cols, 8,
			loadF16, storeF16,
		)

		return
	}

	if cpu.X86.HasSSE2 {
		dispatchReducedMatmulRows(
			MatmulRowFP16SSE2Asm, out, left, right,
			rows, inner, cols, 4,
			loadF16, storeF16,
		)

		return
	}

	runMatmulReduced(out, left, right, rows, inner, cols, loadF16, storeF16)
}

type reducedRowKernelFn func(cRow, aRow, b *uint16, inner, colsBlock, bCols int)

func dispatchReducedMatmulRows(
	kernel reducedRowKernelFn,
	out, left, right unsafe.Pointer,
	rows, inner, cols, align int,
	load reducedLoadFunc,
	store reducedStoreFunc,
) {
	colsBlock := cols &^ (align - 1)

	if colsBlock > 0 {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			rowOffset := rowIndex * cols
			innerOffset := rowIndex * inner

			kernel(
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
		load, store,
		colsBlock,
	)
}
