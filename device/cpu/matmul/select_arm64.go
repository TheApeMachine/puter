//go:build arm64

package matmul

func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	clearFloat32Matrix(out, rows, cols)

	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	colsBlock := cols & ^3
	tailStart := colsBlock

	if colsBlock > 0 {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			MatmulRowFloat32NEONAsm(
				&out[rowIndex*cols],
				&left[rowIndex*inner],
				&right[0],
				inner,
				colsBlock,
			)
		}
	}

	if tailStart == cols {
		return
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := tailStart; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] += leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}

func MatmulFloat64Native(out, left, right []float64, rows, inner, cols int) {
	clearFloat64Matrix(out, rows, cols)

	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	colsBlock := cols & ^1
	tailStart := colsBlock

	if colsBlock > 0 {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			MatmulRowFloat64NEONAsm(
				&out[rowIndex*cols],
				&left[rowIndex*inner],
				&right[0],
				inner,
				colsBlock,
			)
		}
	}

	if tailStart == cols {
		return
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := tailStart; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] += leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	for index := range outView {
		outView[index] = 0
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowStart := int(rowPtr[rowIndex])
		rowEnd := int(rowPtr[rowIndex+1])

		if rowStart == rowEnd {
			continue
		}

		outputRow := outView[rowIndex*cols : (rowIndex+1)*cols]

		for nzIndex := rowStart; nzIndex < rowEnd; nzIndex++ {
			colInLeft := int(colIdx[nzIndex])
			denseRow := rightView[colInLeft*cols : (colInLeft+1)*cols]

			SparseCSRMatMulRowSingleNzNEONAsm(
				&outputRow[0],
				valuesView[nzIndex],
				&denseRow[0],
				cols,
			)
		}
	}
}
