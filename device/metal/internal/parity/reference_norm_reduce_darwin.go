//go:build darwin && cgo

package parity

const normalizationThreadCount = 256

func metalReduceSum(values []float32) float32 {
	reduction := make([]float32, normalizationThreadCount)

	for threadIndex := 0; threadIndex < normalizationThreadCount; threadIndex++ {
		reduction[threadIndex] = metalKahanPartialSum(values, threadIndex)
	}

	return metalTreeReduce256(reduction)
}

func metalKahanPartialSum(values []float32, threadIndex int) float32 {
	localSum := float32(0)
	localCompensation := float32(0)

	for column := threadIndex; column < len(values); column += normalizationThreadCount {
		value := values[column] - localCompensation
		nextSum := localSum + value
		localCompensation = (nextSum - localSum) - value
		localSum = nextSum
	}

	return localSum
}

func metalKahanPartialVariance(values []float32, mean float32, threadIndex int) float32 {
	localVariance := float32(0)
	localCompensation := float32(0)

	for column := threadIndex; column < len(values); column += normalizationThreadCount {
		delta := values[column] - mean
		value := delta*delta - localCompensation
		nextVariance := localVariance + value
		localCompensation = (nextVariance - localVariance) - value
		localVariance = nextVariance
	}

	return localVariance
}

func metalPlainPartialVariance(values []float32, mean float32, threadIndex int) float32 {
	localVariance := float32(0)

	for offset := threadIndex; offset < len(values); offset += normalizationThreadCount {
		delta := values[offset] - mean
		localVariance += delta * delta
	}

	return localVariance
}

func metalTreeReduce256(reduction []float32) float32 {
	for stride := normalizationThreadCount / 2; stride > 0; stride >>= 1 {
		for index := 0; index < stride; index++ {
			reduction[index] += reduction[index+stride]
		}
	}

	return reduction[0]
}
