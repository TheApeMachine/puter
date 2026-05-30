package geometry

var (
	sumFloat64Kernel = func() func(values []float64) float64 {
		return pickF64ReduceKernel(sumFloat64Funcs)
	}()

	sumOfSquaresFloat64Kernel = func() func(values []float64) float64 {
		return pickF64ReduceKernel(sumOfSquaresFloat64Funcs)
	}()

	dotFloat64Kernel = func() func(left, right []float64) float64 {
		return pickF64DotKernel(dotFloat64Funcs)
	}()

	scaleFloat64Kernel = func() func(destination, source []float64, scale float64) {
		return pickF64ScaleKernel(scaleFloat64Funcs)
	}()

	addScalarFloat64Kernel = func() func(destination, source []float64, offset float64) {
		return pickF64AddScalarKernel(addScalarFloat64Funcs)
	}()

	mulFloat64Kernel = func() func(destination, left, right []float64) {
		return pickF64BinaryKernel(mulFloat64Funcs)
	}()

	addFloat64Kernel = func() func(destination, left, right []float64) {
		return pickF64BinaryKernel(addFloat64Funcs)
	}()

	sqrtFloat64Kernel = func() func(destination, source []float64) {
		return pickF64UnaryKernel(sqrtFloat64Funcs)
	}()

	maxFloat64Kernel = func() func(values []float64) float64 {
		return pickF64ReduceKernel(maxFloat64Funcs)
	}()
)

func vecSum(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	return sumFloat64Kernel(values)
}

func vecSumOfSquares(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	return sumOfSquaresFloat64Kernel(values)
}

func vecDotProduct(left, right []float64) float64 {
	if len(left) == 0 {
		return 0
	}

	return dotFloat64Kernel(left, right)
}

func vecScale(destination, source []float64, scale float64) {
	if len(destination) == 0 {
		return
	}

	scaleFloat64Kernel(destination, source, scale)
}

func vecAddScalar(destination, source []float64, offset float64) {
	if len(destination) == 0 {
		return
	}

	addScalarFloat64Kernel(destination, source, offset)
}

func vecMul(destination, left, right []float64) {
	if len(destination) == 0 {
		return
	}

	mulFloat64Kernel(destination, left, right)
}

func vecAdd(destination, left, right []float64) {
	if len(destination) == 0 {
		return
	}

	addFloat64Kernel(destination, left, right)
}

func vecSqrt(destination, source []float64) {
	if len(destination) == 0 {
		return
	}

	sqrtFloat64Kernel(destination, source)
}

func vecMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	return maxFloat64Kernel(values)
}

func matVec512RowMajor(
	destination *[512]float64,
	matrix *[512][512]float64,
	vector *[512]float64,
) {
	for rowIndex := range 512 {
		destination[rowIndex] = vecDotProduct(matrix[rowIndex][:], vector[:])
	}
}

func matVec512ColMajor(
	destination *[512]float64,
	matrix *[512][512]float64,
	vector *[512]float64,
) {
	var scaledColumn [512]float64

	for columnIndex := range 512 {
		vecScale(scaledColumn[:], matrix[columnIndex][:], vector[columnIndex])

		if columnIndex == 0 {
			copy(destination[:], scaledColumn[:])

			continue
		}

		vecAdd(destination[:], destination[:], scaledColumn[:])
	}
}
