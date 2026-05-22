package metal

import (
	"math"
	"math/bits"
)

// physicsPi matches physics.metal constant float physicsPi.
const physicsPi = float32(3.14159265358979323846)

func fftExpected(realIn []float32, imagIn []float32, inverse bool) ([]float32, []float32) {
	count := len(realIn)
	realOut := append([]float32(nil), realIn...)
	imagOut := append([]float32(nil), imagIn...)

	if count == 0 {
		return realOut, imagOut
	}

	if physicsIsPowerOfTwo(count) {
		fftMetalPower2Reference(realIn, imagIn, realOut, imagOut, inverse)
		return realOut, imagOut
	}

	fftMetalNaiveReference(realIn, imagIn, realOut, imagOut, inverse)

	return realOut, imagOut
}

func physicsIsPowerOfTwo(value int) bool {
	return value > 0 && (value&(value-1)) == 0
}

func metalFFTLog2(count uint32) uint32 {
	bitCount := uint32(0)
	value := count

	for value > 1 {
		value >>= 1
		bitCount++
	}

	return bitCount
}

func fftMetalPower2Reference(
	realIn []float32,
	imagIn []float32,
	realOut []float32,
	imagOut []float32,
	inverse bool,
) {
	count := uint32(len(realIn))
	bitCount := metalFFTLog2(count)

	for index := uint32(0); index < count; index++ {
		reversed := bits.Reverse32(index) >> (32 - bitCount)
		realOut[reversed] = realIn[index]
		imagOut[reversed] = imagIn[index]
	}

	for length := uint32(2); length <= count; length <<= 1 {
		for butterfly := uint32(0); butterfly < count/2; butterfly++ {
			fftMetalStageReference(realOut, imagOut, length, inverse, butterfly)
		}
	}

	if !inverse {
		return
	}

	scale := float32(1.0) / float32(count)

	for index := range realOut {
		realOut[index] *= scale
		imagOut[index] *= scale
	}
}

func fftMetalStageReference(
	realValues []float32,
	imagValues []float32,
	length uint32,
	inverse bool,
	butterfly uint32,
) {
	halfLength := length >> 1
	block := butterfly / halfLength
	offset := butterfly - block*halfLength
	upper := int(block*length + offset)
	lower := upper + int(halfLength)

	sign := float32(-1)
	if inverse {
		sign = 1
	}

	angle := sign * 2 * physicsPi / float32(length)
	stepReal := float32(math.Cos(float64(angle)))
	stepImag := float32(math.Sin(float64(angle)))
	twiddleReal := float32(1)
	twiddleImag := float32(0)

	for step := uint32(0); step < offset; step++ {
		nextReal := twiddleReal*stepReal - twiddleImag*stepImag
		nextImag := twiddleReal*stepImag + twiddleImag*stepReal
		twiddleReal, twiddleImag = nextReal, nextImag
	}

	lowerReal := realValues[lower]
	lowerImag := imagValues[lower]
	tempReal := twiddleReal*lowerReal - twiddleImag*lowerImag
	tempImag := twiddleReal*lowerImag + twiddleImag*lowerReal
	upperReal := realValues[upper]
	upperImag := imagValues[upper]

	realValues[lower] = upperReal - tempReal
	imagValues[lower] = upperImag - tempImag
	realValues[upper] = upperReal + tempReal
	imagValues[upper] = upperImag + tempImag
}

func fftMetalNaiveReference(
	realIn []float32,
	imagIn []float32,
	realOut []float32,
	imagOut []float32,
	inverse bool,
) {
	count := len(realIn)

	for index := range count {
		realOut[index], imagOut[index] = fftMetalNaiveBinReference(
			realIn, imagIn, uint32(count), uint32(index), inverse,
		)
	}
}

func fftMetalNaiveBinReference(
	realIn []float32,
	imagIn []float32,
	count uint32,
	index uint32,
	inverse bool,
) (float32, float32) {
	sign := float32(-1)
	if inverse {
		sign = 1
	}

	var sumReal float32
	var sumImag float32

	for source := uint32(0); source < count; source++ {
		angle := sign * 2 * physicsPi * float32(index) * float32(source) / float32(count)
		cosine := float32(math.Cos(float64(angle)))
		sine := float32(math.Sin(float64(angle)))
		realValue := realIn[source]
		imagValue := imagIn[source]
		sumReal += realValue*cosine - imagValue*sine
		sumImag += realValue*sine + imagValue*cosine
	}

	if inverse {
		scale := float32(1.0) / float32(count)
		sumReal *= scale
		sumImag *= scale
	}

	return sumReal, sumImag
}
