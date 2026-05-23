package active_inference

import "math"

/*
activeInferenceLogExact duplicates math/log.go log() so typed FE assembly can be
verified against a local reference without importing math.Log call semantics from asm.
*/
func activeInferenceLogExact(value float64) float64 {
	const (
		ln2Hi = 6.93147180369123816490e-01
		ln2Lo = 1.90821492927058770002e-10
		l1    = 6.666666666666735130e-01
		l2    = 3.999999999940941908e-01
		l3    = 2.857142874366239149e-01
		l4    = 2.222219843214978396e-01
		l5    = 1.818357216161805012e-01
		l6    = 1.531383769920937332e-01
		l7    = 1.479819860511658591e-01
	)

	switch {
	case math.IsNaN(value) || math.IsInf(value, 1):
		return value
	case value < 0:
		return math.NaN()
	case value == 0:
		return math.Inf(-1)
	}

	fraction, exponent := math.Frexp(value)

	if fraction < math.Sqrt2/2 {
		fraction *= 2
		exponent--
	}

	offset := fraction - 1
	exponentFloat := float64(exponent)
	series := offset / (2 + offset)
	seriesSquared := series * series
	seriesFourth := seriesSquared * seriesSquared
	termOne := seriesSquared * (l1 + seriesFourth*(l3+seriesFourth*(l5+seriesFourth*l7)))
	termTwo := seriesFourth * (l2 + seriesFourth*(l4+seriesFourth*l6))
	remainder := termOne + termTwo
	halfSquare := 0.5 * offset * offset

	return exponentFloat*ln2Hi - ((halfSquare - (series*(halfSquare+remainder) + exponentFloat*ln2Lo)) - offset)
}

func activeInferenceLogF64(value float64) float64 {
	return activeInferenceLogExact(value)
}

func clampActiveInferenceLog(value float64) float64 {
	return activeInferenceLogExact(math.Max(activeInferenceEps, value))
}
