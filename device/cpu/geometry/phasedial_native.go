package geometry

import (
	"math"
	"math/cmplx"
)

func dialSimilarity128Scalar(left, right PhaseDial) float64 {
	if len(left) != len(right) || len(left) == 0 {
		return 0
	}

	var dot complex128

	var normLeft float64

	var normRight float64

	for dimIndex := range left {
		dot += cmplx.Conj(left[dimIndex]) * right[dimIndex]
		realPart, imagPart := real(left[dimIndex]), imag(left[dimIndex])
		normLeft += realPart*realPart + imagPart*imagPart
		realPart, imagPart = real(right[dimIndex]), imag(right[dimIndex])
		normRight += realPart*realPart + imagPart*imagPart
	}

	if normLeft == 0 || normRight == 0 {
		return 0
	}

	return real(dot) / (math.Sqrt(normLeft) * math.Sqrt(normRight))
}

func dialRotate128Scalar(out, in PhaseDial, angleRadians float64) {
	if len(out) != len(in) || len(out) == 0 {
		return
	}

	factor := cmplx.Rect(1.0, angleRadians)

	for dimIndex := range out {
		out[dimIndex] = in[dimIndex] * factor
	}
}

func dialAddPhases128Scalar(dial PhaseDial, cosines, sines []float64) {
	for dimIndex := range dial {
		dial[dimIndex] += complex(cosines[dimIndex], sines[dimIndex])
	}
}

func dialComposeMidpoint128Scalar(left, right PhaseDial) PhaseDial {
	if len(left) != len(right) || len(left) == 0 {
		return nil
	}

	normalizedLeft := left.CopyAndNormalize()
	normalizedRight := right.CopyAndNormalize()
	out := make(PhaseDial, len(left))

	for dimIndex := range out {
		out[dimIndex] = normalizedLeft[dimIndex] + normalizedRight[dimIndex]
	}

	var energy float64

	for dimIndex := range out {
		realPart, imagPart := real(out[dimIndex]), imag(out[dimIndex])
		energy += realPart*realPart + imagPart*imagPart
	}

	if energy == 0 {
		return out
	}

	scale := 1.0 / math.Sqrt(energy)

	for dimIndex := range out {
		realPart, imagPart := real(out[dimIndex]), imag(out[dimIndex])
		out[dimIndex] = complex(realPart*scale, imagPart*scale)
	}

	return out
}
