package geometry

import (
	"math"
	"sort"
	"strconv"
)

const (
	jacobiMaxSweeps = 100
	jacobiTolerance = 1e-14
)

/*
JacobiSVD computes the thin SVD of an m×n matrix (m ≥ n) using a pure Go
two-stage Jacobi eigensolver on AᵀA. Returns U (m×n), singular values Σ
(length n, descending), and V (n×n).
*/
func JacobiSVD(matrix [][]float64, rows, cols int) ([][]float64, []float64, [][]float64, error) {
	if rows < cols {
		return nil, nil, nil, ProcrustesError(
			"JacobiSVD requires rows ≥ cols, got " + formatDimensions(rows, cols),
		)
	}

	ata := crossProductATA(matrix, rows, cols)
	eigenvalues, eigenvectors := symmetricJacobiEigen(ata, cols)

	indices := make([]int, cols)
	for index := 0; index < cols; index++ {
		indices[index] = index
	}

	sort.Slice(indices, func(left, right int) bool {
		return eigenvalues[indices[left]] > eigenvalues[indices[right]]
	})

	singularValues := make([]float64, cols)
	rightSingular := make([][]float64, cols)

	for rank := 0; rank < cols; rank++ {
		sourceIndex := indices[rank]
		singularValues[rank] = math.Sqrt(math.Max(0, eigenvalues[sourceIndex]))
		rightSingular[rank] = make([]float64, cols)

		for column := 0; column < cols; column++ {
			rightSingular[rank][column] = eigenvectors[column][sourceIndex]
		}
	}

	leftSingular := make([][]float64, rows)
	for row := 0; row < rows; row++ {
		leftSingular[row] = make([]float64, cols)
	}

	for rank := 0; rank < cols; rank++ {
		sigma := singularValues[rank]

		if sigma <= jacobiTolerance {
			continue
		}

		for row := 0; row < rows; row++ {
			var sum float64

			for column := 0; column < cols; column++ {
				sum += matrix[row][column] * rightSingular[rank][column]
			}

			leftSingular[row][rank] = sum / sigma
		}
	}

	rightFactor := make([][]float64, cols)
	for row := 0; row < cols; row++ {
		rightFactor[row] = make([]float64, cols)

		for column := 0; column < cols; column++ {
			rightFactor[row][column] = rightSingular[column][row]
		}
	}

	return leftSingular, singularValues, rightFactor, nil
}

func crossProductATA(matrix [][]float64, rows, cols int) [][]float64 {
	ata := make([][]float64, cols)

	for row := 0; row < cols; row++ {
		ata[row] = make([]float64, cols)
	}

	for left := 0; left < cols; left++ {
		for right := left; right < cols; right++ {
			var sum float64

			for sample := 0; sample < rows; sample++ {
				sum += matrix[sample][left] * matrix[sample][right]
			}

			ata[left][right] = sum
			ata[right][left] = sum
		}
	}

	return ata
}

func symmetricJacobiEigen(matrix [][]float64, dimension int) ([]float64, [][]float64) {
	working := copySquareMatrix(matrix, dimension)
	eigenvectors := eye(dimension)

	for sweep := 0; sweep < jacobiMaxSweeps; sweep++ {
		pivotRow, pivotColumn, maxOffDiagonal := largestOffDiagonal(working, dimension)

		if maxOffDiagonal < jacobiTolerance {
			break
		}

		applyJacobiRotation(working, eigenvectors, dimension, pivotRow, pivotColumn)
	}

	eigenvalues := make([]float64, dimension)

	for index := 0; index < dimension; index++ {
		eigenvalues[index] = working[index][index]
	}

	return eigenvalues, eigenvectors
}

func largestOffDiagonal(matrix [][]float64, dimension int) (int, int, float64) {
	pivotRow := 0
	pivotColumn := 1
	maxOffDiagonal := math.Abs(matrix[0][1])

	for row := 0; row < dimension; row++ {
		for column := row + 1; column < dimension; column++ {
			offDiagonal := math.Abs(matrix[row][column])

			if offDiagonal > maxOffDiagonal {
				maxOffDiagonal = offDiagonal
				pivotRow = row
				pivotColumn = column
			}
		}
	}

	return pivotRow, pivotColumn, maxOffDiagonal
}

func applyJacobiRotation(
	matrix, eigenvectors [][]float64,
	dimension, pivotRow, pivotColumn int,
) {
	diagonalLeft := matrix[pivotRow][pivotRow]
	diagonalRight := matrix[pivotColumn][pivotColumn]
	offDiagonal := matrix[pivotRow][pivotColumn]

	rotationAngle := 0.5 * math.Atan2(2*offDiagonal, diagonalRight-diagonalLeft)
	cosine := math.Cos(rotationAngle)
	sine := math.Sin(rotationAngle)

	for index := 0; index < dimension; index++ {
		if index == pivotRow || index == pivotColumn {
			continue
		}

		leftEntry := matrix[index][pivotRow]
		rightEntry := matrix[index][pivotColumn]
		matrix[index][pivotRow] = cosine*leftEntry - sine*rightEntry
		matrix[pivotRow][index] = matrix[index][pivotRow]
		matrix[index][pivotColumn] = sine*leftEntry + cosine*rightEntry
		matrix[pivotColumn][index] = matrix[index][pivotColumn]
	}

	matrix[pivotRow][pivotRow] = cosine*cosine*diagonalLeft -
		2*sine*cosine*offDiagonal +
		sine*sine*diagonalRight
	matrix[pivotColumn][pivotColumn] = sine*sine*diagonalLeft +
		2*sine*cosine*offDiagonal +
		cosine*cosine*diagonalRight
	matrix[pivotRow][pivotColumn] = 0
	matrix[pivotColumn][pivotRow] = 0

	for row := 0; row < dimension; row++ {
		leftEntry := eigenvectors[row][pivotRow]
		rightEntry := eigenvectors[row][pivotColumn]
		eigenvectors[row][pivotRow] = cosine*leftEntry - sine*rightEntry
		eigenvectors[row][pivotColumn] = sine*leftEntry + cosine*rightEntry
	}
}

func copySquareMatrix(matrix [][]float64, dimension int) [][]float64 {
	copied := make([][]float64, dimension)

	for row := 0; row < dimension; row++ {
		copied[row] = make([]float64, dimension)
		copy(copied[row], matrix[row])
	}

	return copied
}

func formatDimensions(rows, cols int) string {
	return strconv.Itoa(rows) + " × " + strconv.Itoa(cols)
}
