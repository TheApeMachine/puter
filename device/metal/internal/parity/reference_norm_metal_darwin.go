//go:build darwin && cgo

package parity

import "github.com/theapemachine/manifesto/dtype"

func metalLayerNormRow(
	rowInput, rowOutput, scale, bias []float32,
	storageDType dtype.DType,
) {
	columnCount := len(rowInput)
	roundedInput := make([]float32, columnCount)
	roundedScale := make([]float32, len(scale))
	roundedBias := make([]float32, len(bias))

	for columnIndex := range roundedInput {
		roundedInput[columnIndex] = metalStorageLoad(rowInput[columnIndex], storageDType)
	}

	for columnIndex := range roundedScale {
		roundedScale[columnIndex] = metalStorageLoad(scale[columnIndex], storageDType)
		roundedBias[columnIndex] = metalStorageLoad(bias[columnIndex], storageDType)
	}

	mean := metalReduceSum(roundedInput) / float32(columnCount)

	reduction := make([]float32, normalizationThreadCount)

	for threadIndex := 0; threadIndex < normalizationThreadCount; threadIndex++ {
		reduction[threadIndex] = metalKahanPartialVariance(roundedInput, mean, threadIndex)
	}

	varianceSum := metalTreeReduce256(reduction)
	invStdDev := metalFloat32InvStdDev(varianceSum, columnCount)

	for columnIndex := 0; columnIndex < columnCount; columnIndex++ {
		loaded := roundedInput[columnIndex]
		normalized := (loaded - mean) * invStdDev
		stored := metalStorageStore(
			normalized*roundedScale[columnIndex]+roundedBias[columnIndex],
			storageDType,
		)
		rowOutput[columnIndex] = stored
	}
}

func metalGroupNormGroup(
	groupInput, groupOutput []float32,
	scale, bias []float32,
	channelsPerGroup, spatial int,
	storageDType dtype.DType,
) {
	groupSize := len(groupInput)
	roundedInput := make([]float32, groupSize)
	roundedScale := make([]float32, len(scale))
	roundedBias := make([]float32, len(bias))

	for offset := range roundedInput {
		roundedInput[offset] = metalStorageLoad(groupInput[offset], storageDType)
	}

	for channelIndex := range roundedScale {
		roundedScale[channelIndex] = metalStorageLoad(scale[channelIndex], storageDType)
		roundedBias[channelIndex] = metalStorageLoad(bias[channelIndex], storageDType)
	}

	mean := metalReduceSum(roundedInput) / float32(groupSize)

	reduction := make([]float32, normalizationThreadCount)

	for threadIndex := 0; threadIndex < normalizationThreadCount; threadIndex++ {
		reduction[threadIndex] = metalPlainPartialVariance(roundedInput, mean, threadIndex)
	}

	varianceSum := metalTreeReduce256(reduction)
	invStdDev := metalFloat32InvStdDev(varianceSum, groupSize)

	for offset := 0; offset < groupSize; offset++ {
		channelIndex := offset / spatial
		normalized := (roundedInput[offset] - mean) * invStdDev
		stored := metalStorageStore(
			normalized*roundedScale[channelIndex]+roundedBias[channelIndex],
			storageDType,
		)
		groupOutput[offset] = stored
	}
}
