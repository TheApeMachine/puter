//go:build !amd64 && !arm64

package matmul

func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	matmulFloat32Scalar(out, left, right, rows, inner, cols)
}

func MatmulFloat64Native(out, left, right []float64, rows, inner, cols int) {
	matmulFloat64Scalar(out, left, right, rows, inner, cols)
}

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	SparseCSRMatMulFloat32Scalar(
		outView, valuesView, rightView,
		rowPtr, colIdx,
		rows, cols,
	)
}

func matmulFloat32Scalar(out, left, right []float32, rows, inner, cols int) {
	clearFloat32Matrix(out, rows, cols)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] += leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}

func matmulFloat64Scalar(out, left, right []float64, rows, inner, cols int) {
	clearFloat64Matrix(out, rows, cols)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] +=
					leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}
