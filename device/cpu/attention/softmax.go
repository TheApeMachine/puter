package attention

import "math"

func stableSoftmaxRow(scores []float32) {
	if len(scores) == 0 {
		return
	}

	maximum := ReduceMaxFloat32Native(scores)

	if normalizeInfiniteSoftmaxRow(scores, maximum) {
		return
	}

	sum := SoftmaxRowFillExpsNative(scores, scores, maximum)
	normalizeSoftmaxRow(scores, sum)
}

func SoftmaxRowFillExpsNative(dst, src []float32, maximum float32) float32 {
	var sum float32

	for index, value := range src {
		shifted := float32(math.Exp(float64(value - maximum)))
		dst[index] = shifted
		sum += shifted
	}

	return sum
}

func normalizeInfiniteSoftmaxRow(scores []float32, maximum float32) bool {
	if !math.IsInf(float64(maximum), 0) {
		return false
	}

	var positiveInfinityCount int

	if math.IsInf(float64(maximum), 1) {
		for _, score := range scores {
			if math.IsInf(float64(score), 1) {
				positiveInfinityCount++
			}
		}
	}

	if math.IsInf(float64(maximum), -1) {
		for index := range scores {
			scores[index] = 0
		}

		return true
	}

	if positiveInfinityCount == 0 {
		for index := range scores {
			scores[index] = 0
		}

		return true
	}

	weight := 1 / float32(positiveInfinityCount)

	for index, score := range scores {
		if math.IsInf(float64(score), 1) {
			scores[index] = weight
			continue
		}

		scores[index] = 0
	}

	return true
}

func normalizeSoftmaxRow(row []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index := range row {
		row[index] /= sum
	}
}
