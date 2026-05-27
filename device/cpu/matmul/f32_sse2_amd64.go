//go:build amd64

package matmul

//go:noescape
func MatmulRowFloat32SSE2Asm(cRow, aRow, b *float32, inner, cols int)

func MatmulFloat32SSE2(out, left, right []float32, rows, inner, cols int) {
	clearFloat32Matrix(out, rows, cols)

	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		MatmulRowFloat32SSE2Asm(
			&out[rowIndex*cols],
			&left[rowIndex*inner],
			&right[0],
			inner,
			cols,
		)
	}
}
