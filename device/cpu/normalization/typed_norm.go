package normalization

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func runGroupNormBF16(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.BF16)(scale), channels)
	biasView := unsafe.Slice((*dtype.BF16)(bias), channels)
	outputView := unsafe.Slice((*dtype.BF16)(output), batch*channels*spatial)
	groupNormSlicesBF16(config, inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runGroupNormF16(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.F16)(scale), channels)
	biasView := unsafe.Slice((*dtype.F16)(bias), channels)
	outputView := unsafe.Slice((*dtype.F16)(output), batch*channels*spatial)
	groupNormSlicesF16(config, inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runInstanceNormBF16(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.BF16)(scale), channels)
	biasView := unsafe.Slice((*dtype.BF16)(bias), channels)
	outputView := unsafe.Slice((*dtype.BF16)(output), batch*channels*spatial)
	instanceNormSlicesBF16(inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runInstanceNormF16(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.F16)(scale), channels)
	biasView := unsafe.Slice((*dtype.F16)(bias), channels)
	outputView := unsafe.Slice((*dtype.F16)(output), batch*channels*spatial)
	instanceNormSlicesF16(inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runBatchNormEvalBF16(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.BF16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.BF16)(scale), channels)
	biasView := unsafe.Slice((*dtype.BF16)(bias), channels)
	meanView := unsafe.Slice((*dtype.BF16)(mean), channels)
	varianceView := unsafe.Slice((*dtype.BF16)(variance), channels)
	outputView := unsafe.Slice((*dtype.BF16)(output), batch*channels*spatial)
	batchNormEvalSlicesBF16(
		inputView, scaleView, biasView, meanView, varianceView, outputView,
		batch, channels, spatial,
	)
}

func runBatchNormEvalF16(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*dtype.F16)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*dtype.F16)(scale), channels)
	biasView := unsafe.Slice((*dtype.F16)(bias), channels)
	meanView := unsafe.Slice((*dtype.F16)(mean), channels)
	varianceView := unsafe.Slice((*dtype.F16)(variance), channels)
	outputView := unsafe.Slice((*dtype.F16)(output), batch*channels*spatial)
	batchNormEvalSlicesF16(
		inputView, scaleView, biasView, meanView, varianceView, outputView,
		batch, channels, spatial,
	)
}

func groupNormSlicesBF16(
	config GroupNormConfig,
	input, scale, bias, output []dtype.BF16,
	batch, channels, spatial int,
) {
	channelsPerGroup := channels / config.Groups
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < config.Groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial
			normalizeGroupBF16(
				input[groupStart:groupStart+groupSize],
				output[groupStart:groupStart+groupSize],
				scale[channelStart:channelStart+channelsPerGroup],
				bias[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}
}

func groupNormSlicesF16(
	config GroupNormConfig,
	input, scale, bias, output []dtype.F16,
	batch, channels, spatial int,
) {
	channelsPerGroup := channels / config.Groups
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < config.Groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial
			normalizeGroupF16(
				input[groupStart:groupStart+groupSize],
				output[groupStart:groupStart+groupSize],
				scale[channelStart:channelStart+channelsPerGroup],
				bias[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}
}

func normalizeGroupBF16(
	inputSlice, outSlice, scaleSlice, biasSlice []dtype.BF16,
	channelsPerGroup, spatial int,
) {
	elementCount := len(inputSlice)
	meanValue := reduction.SumBFloat16Native(inputSlice)
	mean := (&meanValue).Float32() / float32(elementCount)
	variance := normVarianceBF16(inputSlice, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance+normEpsilon)))

	for channelIndex := 0; channelIndex < channelsPerGroup; channelIndex++ {
		channelStart := channelIndex * spatial
		row := inputSlice[channelStart : channelStart+spatial]
		outRow := outSlice[channelStart : channelStart+spatial]
		scaleValue := (&scaleSlice[channelIndex]).Float32()
		biasValue := (&biasSlice[channelIndex]).Float32()
		applyNormRowBF16(outRow, row, mean, invStdDev, scaleValue, biasValue)
	}
}

func normalizeGroupF16(
	inputSlice, outSlice, scaleSlice, biasSlice []dtype.F16,
	channelsPerGroup, spatial int,
) {
	elementCount := len(inputSlice)
	meanValue := reduction.SumFloat16Native(inputSlice)
	mean := meanValue.Float32() / float32(elementCount)
	variance := normVarianceF16(inputSlice, mean)
	invStdDev := float32(1.0 / math.Sqrt(float64(variance+normEpsilon)))

	for channelIndex := 0; channelIndex < channelsPerGroup; channelIndex++ {
		channelStart := channelIndex * spatial
		row := inputSlice[channelStart : channelStart+spatial]
		outRow := outSlice[channelStart : channelStart+spatial]
		applyNormRowF16(outRow, row, mean, invStdDev, scaleSlice[channelIndex].Float32(), biasSlice[channelIndex].Float32())
	}
}

func instanceNormSlicesBF16(input, scale, bias, output []dtype.BF16, batch, channels, spatial int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]
			meanValue := reduction.SumBFloat16Native(row)
			mean := (&meanValue).Float32() / float32(spatial)
			variance := normVarianceBF16(row, mean)
			invStdDev := float32(1.0 / math.Sqrt(float64(variance+normEpsilon)))
			applyNormRowBF16(
				outRow, row, mean, invStdDev,
				(&scale[channelIndex]).Float32(),
				(&bias[channelIndex]).Float32(),
			)
		}
	}
}

func instanceNormSlicesF16(input, scale, bias, output []dtype.F16, batch, channels, spatial int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]
			meanValue := reduction.SumFloat16Native(row)
			mean := meanValue.Float32() / float32(spatial)
			variance := normVarianceF16(row, mean)
			invStdDev := float32(1.0 / math.Sqrt(float64(variance+normEpsilon)))
			applyNormRowF16(
				outRow, row, mean, invStdDev,
				scale[channelIndex].Float32(),
				bias[channelIndex].Float32(),
			)
		}
	}
}

func batchNormEvalSlicesBF16(
	input, scale, bias, mean, variance, output []dtype.BF16,
	batch, channels, spatial int,
) {
	for channelIndex := 0; channelIndex < channels; channelIndex++ {
		invStdDev := float32(1.0 / math.Sqrt(float64((&variance[channelIndex]).Float32()+normEpsilon)))
		channelMean := (&mean[channelIndex]).Float32()
		channelScale := (&scale[channelIndex]).Float32()
		channelBias := (&bias[channelIndex]).Float32()

		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]
			applyNormRowBF16(outRow, row, channelMean, invStdDev, channelScale, channelBias)
		}
	}
}

func batchNormEvalSlicesF16(
	input, scale, bias, mean, variance, output []dtype.F16,
	batch, channels, spatial int,
) {
	for channelIndex := 0; channelIndex < channels; channelIndex++ {
		invStdDev := float32(1.0 / math.Sqrt(float64(variance[channelIndex].Float32()+normEpsilon)))
		channelMean := mean[channelIndex].Float32()
		channelScale := scale[channelIndex].Float32()
		channelBias := bias[channelIndex].Float32()

		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]
			applyNormRowF16(outRow, row, channelMean, invStdDev, channelScale, channelBias)
		}
	}
}

func applyNormRowBF16(
	outRow, row []dtype.BF16,
	mean, invStdDev, scaleValue, biasValue float32,
) {
	combined := make([]dtype.BF16, len(row))

	for index := range row {
		normalized := ((&row[index]).Float32() - mean) * invStdDev
		combined[index] = dtype.NewBfloat16FromFloat32(normalized*scaleValue + biasValue)
	}

	copy(outRow, combined)
}

func applyNormRowF16(
	outRow, row []dtype.F16,
	mean, invStdDev, scaleValue, biasValue float32,
) {
	combined := make([]dtype.F16, len(row))

	for index := range row {
		normalized := (row[index].Float32() - mean) * invStdDev
		combined[index] = dtype.Fromfloat32(normalized*scaleValue + biasValue)
	}

	copy(outRow, combined)
}

func normVarianceBF16(row []dtype.BF16, mean float32) float32 {
	var variance float32

	for index := range row {
		delta := (&row[index]).Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}

func normVarianceF16(row []dtype.F16, mean float32) float32 {
	var variance float32

	for index := range row {
		delta := row[index].Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}
