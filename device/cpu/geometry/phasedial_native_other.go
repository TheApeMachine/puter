//go:build !arm64

package geometry

import "math"

func dialNormalize128Native(dial PhaseDial) {
	var sumSq float64

	for _, val := range dial {
		realPart, imagPart := real(val), imag(val)
		sumSq += realPart*realPart + imagPart*imagPart
	}

	if sumSq == 0 {
		return
	}

	inv := 1.0 / math.Sqrt(sumSq)

	for index := range dial {
		realPart, imagPart := real(dial[index]), imag(dial[index])
		dial[index] = complex(realPart*inv, imagPart*inv)
	}
}

func dialSimilarity128Native(left, right PhaseDial) float64 {
	return dialSimilarity128Scalar(left, right)
}

func dialRotate128Native(out, in PhaseDial, angleRadians float64) {
	dialRotate128Scalar(out, in, angleRadians)
}

func dialAddPhases128Native(dial PhaseDial, cosines, sines []float64) {
	dialAddPhases128Scalar(dial, cosines, sines)
}

func dialComposeMidpoint128Native(left, right PhaseDial) PhaseDial {
	return dialComposeMidpoint128Scalar(left, right)
}
