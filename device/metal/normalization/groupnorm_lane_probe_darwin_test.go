//go:build darwin && cgo

package normalization

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpunormalization "github.com/theapemachine/puter/device/cpu/normalization"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	cpureduction "github.com/theapemachine/puter/device/cpu/reduction"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGroupNormLane44Probe(t *testing.T) {
	batch := 2
	channels := 8
	groups := 2
	spatial := 7
	elementCount := batch * channels * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(elementCount, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)
	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.Float32)

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float32)
	scaleTensor := harness.UploadVector(scale, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	if err := DispatchGroupNormRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		dtype.Float32,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch groupnorm: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)

	laneIndex := 44
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial

	batchIndex := laneIndex / (channels * spatial)
	channelIndex := (laneIndex / spatial) % channels
	groupIndex := channelIndex / channelsPerGroup
	channelStart := groupIndex * channelsPerGroup
	groupOffset := (batchIndex*channels + channelStart) * spatial
	groupSlice := input[groupOffset : groupOffset+groupSize]

	sum := cpureduction.SumFloat32Native(groupSlice)
	mean := float32(float64(sum) / float64(groupSize))
	varianceSum := cpunormalization.NormSquaredDiffSumNative(groupSlice, mean)
	varianceSumGeneric := cpunormalization.NormSquaredDiffSumGeneric(groupSlice, mean)
	variance := float64(varianceSum) / float64(groupSize)
	invStdDev := float32(1.0 / math.Sqrt(variance+1e-5))

	scaleValue := scale[channelIndex]
	biasValue := bias[channelIndex]
	inputValue := input[laneIndex]

	manual := (inputValue-mean)*invStdDev*scaleValue + biasValue
	manualFMA := scaleValue*(inputValue-mean)*invStdDev + biasValue
	effectiveInv := ((got[laneIndex] - biasValue) / scaleValue) / (inputValue - mean)

	normalized := (inputValue - mean) * invStdDev
	genericOut := normalized*scaleValue + biasValue

	nativeOut := make([]float32, groupSize)
	cpunormalization.NormApplyConstScaleBiasNative(
		nativeOut,
		groupSlice,
		mean,
		invStdDev,
		scaleValue,
		biasValue,
	)

	t.Logf(
		"lane %d mean=%.9g inv=%.9g effectiveInv=%.9g varSumNative=%.9g varSumGeneric=%.9g chain=%.9g generic=%.9g fma=%.9g native=%.9g want=%.9g metal=%.9g nativeUlp=%d genericUlp=%d metalUlp=%d",
		laneIndex,
		mean, invStdDev, effectiveInv, varianceSum, varianceSumGeneric,
		manual, genericOut, manualFMA, nativeOut[laneIndex-groupOffset], want[laneIndex], got[laneIndex],
		cpuparity.Float32ULPDistance(nativeOut[laneIndex-groupOffset], want[laneIndex]),
		cpuparity.Float32ULPDistance(genericOut, want[laneIndex]),
		cpuparity.Float32ULPDistance(got[laneIndex], want[laneIndex]),
	)
}
