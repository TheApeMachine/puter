package geometry

import "unsafe"

/*
geometricProductFloat64Scalar is the reference PGA geometric product for
even-subalgebra multivectors (8 float64 components).
*/
func geometricProductFloat64Scalar(left, right, destination *float64) {
	leftValues := multivectorView(left)
	rightValues := multivectorView(right)
	destinationValues := multivectorView(destination)

	destinationValues[0] = leftValues[0]*rightValues[0] -
		leftValues[4]*rightValues[4] -
		leftValues[5]*rightValues[5] -
		leftValues[6]*rightValues[6]

	destinationValues[1] = leftValues[0]*rightValues[1] +
		leftValues[1]*rightValues[0] -
		leftValues[2]*rightValues[4] +
		leftValues[3]*rightValues[5] +
		leftValues[4]*rightValues[2] -
		leftValues[5]*rightValues[3] -
		leftValues[6]*rightValues[7] -
		leftValues[7]*rightValues[6]

	destinationValues[2] = leftValues[0]*rightValues[2] +
		leftValues[1]*rightValues[4] +
		leftValues[2]*rightValues[0] -
		leftValues[3]*rightValues[6] -
		leftValues[4]*rightValues[1] -
		leftValues[5]*rightValues[7] +
		leftValues[6]*rightValues[3] -
		leftValues[7]*rightValues[5]

	destinationValues[3] = leftValues[0]*rightValues[3] -
		leftValues[1]*rightValues[5] +
		leftValues[2]*rightValues[6] +
		leftValues[3]*rightValues[0] -
		leftValues[4]*rightValues[7] +
		leftValues[5]*rightValues[1] -
		leftValues[6]*rightValues[2] -
		leftValues[7]*rightValues[4]

	destinationValues[4] = leftValues[0]*rightValues[4] +
		leftValues[4]*rightValues[0] +
		leftValues[5]*rightValues[6] -
		leftValues[6]*rightValues[5]

	destinationValues[5] = leftValues[0]*rightValues[5] -
		leftValues[4]*rightValues[6] +
		leftValues[5]*rightValues[0] +
		leftValues[6]*rightValues[4]

	destinationValues[6] = leftValues[0]*rightValues[6] +
		leftValues[4]*rightValues[5] -
		leftValues[5]*rightValues[4] +
		leftValues[6]*rightValues[0]

	destinationValues[7] = leftValues[0]*rightValues[7] +
		leftValues[1]*rightValues[6] +
		leftValues[2]*rightValues[5] +
		leftValues[3]*rightValues[4] +
		leftValues[4]*rightValues[3] +
		leftValues[5]*rightValues[2] +
		leftValues[6]*rightValues[1] +
		leftValues[7]*rightValues[0]
}

func multivectorView(base *float64) *[8]float64 {
	return (*[8]float64)(unsafe.Pointer(base))
}

/*
rotorSimilarity128Scalar averages scalar parts of left[k]·reverse(right[k]).
*/
func rotorSimilarity128Scalar(left, right *float64, count int) float64 {
	if count == 0 {
		return 0
	}

	var dotSum float64
	var reversed [8]float64

	for rotorIndex := range count {
		leftRotor := rotorOffsetPointer(left, rotorIndex)
		rightRotor := rotorOffsetPointer(right, rotorIndex)

		reverseMultivectorFloat64(rightRotor, &reversed[0])

		var product [8]float64

		geometricProductFloat64Scalar(leftRotor, &reversed[0], &product[0])

		dotSum += product[0]
	}

	return dotSum / float64(count)
}

func reverseMultivectorFloat64(source, destination *float64) {
	sourceValues := multivectorView(source)
	destinationValues := multivectorView(destination)

	destinationValues[0] = sourceValues[0]
	destinationValues[1] = -sourceValues[1]
	destinationValues[2] = -sourceValues[2]
	destinationValues[3] = -sourceValues[3]
	destinationValues[4] = -sourceValues[4]
	destinationValues[5] = -sourceValues[5]
	destinationValues[6] = -sourceValues[6]
	destinationValues[7] = sourceValues[7]
}

func rotorOffsetPointer(base *float64, rotorIndex int) *float64 {
	byteOffset := uintptr(rotorIndex*8) * unsafe.Sizeof(float64(0))

	return (*float64)(unsafe.Add(unsafe.Pointer(base), byteOffset))
}
