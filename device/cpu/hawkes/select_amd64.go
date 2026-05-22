//go:build amd64

package hawkes

import (
	"math"

	"golang.org/x/sys/cpu"
)

func hawkesExpSumNative(exponents []float32, count int) float32 {
	if count <= 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		asmCount := count &^ 15

		if asmCount > 0 {
			sum := HawkesExpSumFloat32AVX512Asm(&exponents[0], asmCount)
			for index := asmCount; index < count; index++ {
				sum += hawkesExpScalar(exponents[index])
			}

			return sum
		}
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		asmCount := count &^ 7

		if asmCount > 0 {
			sum := HawkesExpSumFloat32AVX2Asm(&exponents[0], asmCount)
			for index := asmCount; index < count; index++ {
				sum += hawkesExpScalar(exponents[index])
			}

			return sum
		}
	}

	if cpu.X86.HasSSE2 {
		asmCount := count &^ 3

		if asmCount > 0 {
			sum := HawkesExpSumFloat32SSE2Asm(&exponents[0], asmCount)
			for index := asmCount; index < count; index++ {
				sum += hawkesExpScalar(exponents[index])
			}

			return sum
		}
	}

	sum := float32(0)

	for index := 0; index < count; index++ {
		sum += hawkesExpScalar(exponents[index])
	}

	return sum
}

func hawkesScaledExpStoreNative(
	exponents []float32,
	alpha float32,
	out []float32,
	count int,
) {
	if count <= 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		asmCount := count &^ 15

		if asmCount > 0 {
			HawkesScaledExpStoreFloat32AVX512Asm(&exponents[0], alpha, &out[0], asmCount)
			for index := asmCount; index < count; index++ {
				out[index] = alpha * hawkesExpScalar(exponents[index])
			}

			return
		}
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		asmCount := count &^ 7

		if asmCount > 0 {
			HawkesScaledExpStoreFloat32AVX2Asm(&exponents[0], alpha, &out[0], asmCount)
			for index := asmCount; index < count; index++ {
				out[index] = alpha * hawkesExpScalar(exponents[index])
			}

			return
		}
	}

	if cpu.X86.HasSSE2 {
		asmCount := count &^ 3

		if asmCount > 0 {
			HawkesScaledExpStoreFloat32SSE2Asm(&exponents[0], alpha, &out[0], asmCount)
			for index := asmCount; index < count; index++ {
				out[index] = alpha * hawkesExpScalar(exponents[index])
			}

			return
		}
	}

	for index := 0; index < count; index++ {
		out[index] = alpha * hawkesExpScalar(exponents[index])
	}
}

func HawkesIntensityNative(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	scratch := BorrowFloat32Buffer(len(eventTimes))
	defer ReleaseFloat32Buffer(scratch)

	for queryIndex, queryTime := range queryTimes {
		out[queryIndex] = mu + hawkesExcitationAtNative(
			queryTime, eventTimes, scratch, alpha, beta,
		)
	}
}

func hawkesExcitationAtNative(
	queryTime float32,
	eventTimes, scratch []float32,
	alpha, beta float32,
) float32 {
	validCount := 0

	for _, eventTime := range eventTimes {
		if eventTime > queryTime {
			continue
		}

		scratch[validCount] = -beta * (queryTime - eventTime)
		validCount++
	}

	if validCount == 0 {
		return 0
	}

	return alpha * hawkesExpSumNative(scratch[:validCount], validCount)
}

func HawkesKernelMatrixNative(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	eventCount := len(eventTimes)
	scratch := BorrowFloat32Buffer(eventCount)
	defer ReleaseFloat32Buffer(scratch)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		rowStart := rowIndex * eventCount

		for colIndex := rowIndex; colIndex < eventCount; colIndex++ {
			out[rowStart+colIndex] = 0
		}

		if rowIndex == 0 {
			continue
		}

		for colIndex := 0; colIndex < rowIndex; colIndex++ {
			scratch[colIndex] = -beta * (eventTimes[rowIndex] - eventTimes[colIndex])
		}

		hawkesScaledExpStoreNative(scratch[:rowIndex], alpha, out[rowStart:rowStart+rowIndex], rowIndex)
	}
}

func HawkesLogLikelihoodNative(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	eventCount := len(eventTimes)
	scratch := BorrowFloat32Buffer(eventCount)
	defer ReleaseFloat32Buffer(scratch)

	var sumLog float64

	for eventIndex := range eventTimes {
		validCount := 0

		for previousIndex := 0; previousIndex < eventIndex; previousIndex++ {
			delta := eventTimes[eventIndex] - eventTimes[previousIndex]
			scratch[validCount] = -beta * delta
			validCount++
		}

		intensity := mu

		if validCount > 0 {
			intensity += alpha * hawkesExpSumNative(scratch[:validCount], validCount)
		}

		sumLog += math.Log(math.Max(1e-12, float64(intensity)))
	}

	compensator := float64(mu * totalT)

	for _, eventTime := range eventTimes {
		compensator += float64(alpha/beta) * (1 - math.Exp(float64(-beta*(totalT-eventTime))))
	}

	out[0] = float32(sumLog - compensator)
}

func MarkovMutualInformationNative(joint []float32, xCount, yCount int, out []float32) {
	MarkovMutualInformationScalar(joint, xCount, yCount, out)
}
