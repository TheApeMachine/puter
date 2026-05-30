package geometry

import "math"

func sumFloat64Scalar(values []float64) float64 {
	var total float64

	for _, value := range values {
		total += value
	}

	return total
}

func sumOfSquaresFloat64Scalar(values []float64) float64 {
	var total float64

	for _, value := range values {
		total += value * value
	}

	return total
}

func dotFloat64Scalar(left, right []float64) float64 {
	var total float64

	for index := range left {
		total += left[index] * right[index]
	}

	return total
}

func scaleFloat64Scalar(destination, source []float64, scale float64) {
	for index := range destination {
		destination[index] = source[index] * scale
	}
}

func addScalarFloat64Scalar(destination, source []float64, offset float64) {
	for index := range destination {
		destination[index] = source[index] + offset
	}
}

func mulFloat64Scalar(destination, left, right []float64) {
	for index := range destination {
		destination[index] = left[index] * right[index]
	}
}

func addFloat64Scalar(destination, left, right []float64) {
	for index := range destination {
		destination[index] = left[index] + right[index]
	}
}

func sqrtFloat64Scalar(destination, source []float64) {
	for index := range destination {
		destination[index] = math.Sqrt(source[index])
	}
}

func maxFloat64Scalar(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	maximum := values[0]

	for index := 1; index < len(values); index++ {
		if values[index] > maximum {
			maximum = values[index]
		}
	}

	return maximum
}
