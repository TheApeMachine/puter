//go:build darwin && cgo

package normalization

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/cpu/reduction"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGroupNormF16StatsProbeSpatial1024Lane12519(t *testing.T) {
	batch := 2
	channels := 8
	groups := 2
	spatial := 1024
	laneIndex := 12519
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(batch*channels*spatial, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)

	inputF16 := make([]dtype.F16, len(input))
	scaleF16 := make([]dtype.F16, len(scale))
	biasF16 := make([]dtype.F16, len(bias))

	for index := range input {
		inputF16[index] = dtype.Fromfloat32(input[index])
	}

	for index := range scale {
		scaleF16[index] = dtype.Fromfloat32(scale[index])
	}

	for index := range bias {
		biasF16[index] = dtype.Fromfloat32(bias[index])
	}

	batchIndex := laneIndex / (channels * spatial)
	channel := (laneIndex % (channels * spatial)) / spatial
	groupIndex := channel / channelsPerGroup
	groupStart := batchIndex*channels*spatial + groupIndex*channelsPerGroup*spatial
	groupSlice := inputF16[groupStart : groupStart+groupSize]
	statsRow := batchIndex*groups + groupIndex

	cpuSum := reduction.SumFloat16Native(groupSlice)
	cpuMean := cpuSum.Float32() / float32(groupSize)
	cpuVariance := groupNormVarianceF16Sim(groupSlice, cpuMean)
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))

	channelInGroup := (laneIndex - groupStart) / spatial
	scaleChannel := groupIndex*channelsPerGroup + channelInGroup
	cpuOut := (inputF16[laneIndex].Float32()-cpuMean)*cpuInv*scaleF16[scaleChannel].Float32() + biasF16[scaleChannel].Float32()

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float16)
	statsTensor := harness.UploadVector(make([]float32, batch*groups*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchGroupNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)
	metalMean := stats[statsRow*2]
	metalInv := stats[statsRow*2+1]
	metalOut := (inputF16[laneIndex].Float32()-metalMean)*metalInv*scaleF16[scaleChannel].Float32() + biasF16[scaleChannel].Float32()

	scaleTensor := harness.UploadVector(scale, dtype.Float16)
	biasTensor := harness.UploadVector(bias, dtype.Float16)
	splitOutputTensor := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer splitOutputTensor.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		splitOutputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.Float16)

	cpuStatsFull := make([]float32, batch*groups*2)
	cpuStatsFull[statsRow*2] = cpuMean
	cpuStatsFull[statsRow*2+1] = cpuInv
	cpuStatsTensorFull := harness.UploadVector(cpuStatsFull, dtype.Float32)
	cpuApplyOutput := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer cpuStatsTensorFull.Close()
	defer cpuApplyOutput.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		cpuApplyOutput.Ref(),
		cpuStatsTensorFull.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch cpu-stats apply: %v", err)
	}

	cpuStatsGot := harness.DownloadFloat32(cpuApplyOutput, dtype.Float16)
	cpuStatsBytes := cpuApplyOutput.ReadBytes()
	t.Logf("cpuStatsApply[%d]=%v bits=0x%04x ulp=%d",
		laneIndex, cpuStatsGot[laneIndex],
		dtype.Frombits(uint16(cpuStatsBytes[laneIndex*2])|uint16(cpuStatsBytes[laneIndex*2+1])<<8).Bits(),
		parity.Float32ULPDistance(cpuStatsGot[laneIndex], want[laneIndex]))

	splitGot := harness.DownloadFloat32(splitOutputTensor, dtype.Float16)

	t.Logf("statsRow=%d groupStart=%d", statsRow, groupStart)
	t.Logf("cpuSum=0x%04x cpuMean=%v metalMean=%v meanUlp=%d",
		uint16(cpuSum), cpuMean, metalMean, parity.Float32ULPDistance(cpuMean, metalMean))
	t.Logf("cpuInv=%v metalInv=%v invUlp=%d",
		cpuInv, metalInv, parity.Float32ULPDistance(cpuInv, metalInv))
	t.Logf("cpuOut=%v metalOut=%v outUlp=%d storedCpu=%v storedMetal=%v",
		cpuOut, metalOut, parity.Float32ULPDistance(metalOut, cpuOut),
		dtype.Fromfloat32(cpuOut).Float32(), dtype.Fromfloat32(metalOut).Float32())

	splitBytes := splitOutputTensor.ReadBytes()
	inputBytes := inputTensor.ReadBytes()
	scaleBytes := scaleTensor.ReadBytes()
	biasBytes := biasTensor.ReadBytes()
	deviceInput := dtype.Frombits(uint16(inputBytes[laneIndex*2]) | uint16(inputBytes[laneIndex*2+1])<<8)
	deviceScale := dtype.Frombits(uint16(scaleBytes[scaleChannel*2]) | uint16(scaleBytes[scaleChannel*2+1])<<8)
	deviceBias := dtype.Frombits(uint16(biasBytes[scaleChannel*2]) | uint16(biasBytes[scaleChannel*2+1])<<8)
	deviceOut := (deviceInput.Float32()-metalMean)*metalInv*deviceScale.Float32() + deviceBias.Float32()
	t.Logf("deviceInput=0x%04x hostInput=0x%04x scale=0x%04x bias=0x%04x",
		deviceInput.Bits(), inputF16[laneIndex].Bits(), deviceScale.Bits(), deviceBias.Bits())
	t.Logf("deviceManualOut=%v deviceStored=0x%04x",
		deviceOut, dtype.Fromfloat32(deviceOut).Bits())
	t.Logf("wantBits=0x%04x splitBits=0x%04x cpuBits=0x%04x",
		dtype.Fromfloat32(want[laneIndex]).Bits(),
		dtype.Frombits(uint16(splitBytes[laneIndex*2])|uint16(splitBytes[laneIndex*2+1])<<8).Bits(),
		dtype.Fromfloat32(cpuOut).Bits())
	t.Logf("want[%d]=%v", laneIndex, want[laneIndex])
	t.Logf("splitGot[%d]=%v splitUlp=%d", laneIndex, splitGot[laneIndex], parity.Float32ULPDistance(splitGot[laneIndex], want[laneIndex]))

	outputTensor := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer outputTensor.Close()

	if err := DispatchGroupNormRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch groupnorm: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float16)
	t.Logf("got[%d]=%v gotUlp=%d", laneIndex, got[laneIndex], parity.Float32ULPDistance(got[laneIndex], want[laneIndex]))
}

func groupNormVarianceF16Sim(groupSlice []dtype.F16, mean float32) float32 {
	var variance float32

	for index := range groupSlice {
		delta := groupSlice[index].Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(groupSlice))
}

func simulateMetalF16SumNative(values []dtype.F16) dtype.F16 {
	count := len(values)
	var acc [4]float32
	index := 0

	for index+16 <= count {
		var partial [8]dtype.F16

		for lane := 0; lane < 8; lane++ {
			partial[lane] = values[index+lane]
		}

		for lane := 0; lane < 8; lane++ {
			partial[lane] = dtype.Fromfloat32(values[index+8+lane].Float32() + partial[lane].Float32())
		}

		for lane := 0; lane < 4; lane++ {
			acc[0] += partial[lane].Float32()
		}

		for lane := 4; lane < 8; lane++ {
			acc[1] += partial[lane].Float32()
		}

		index += 16
	}

	for index+8 <= count {
		for lane := 0; lane < 4; lane++ {
			acc[0] += values[index+lane].Float32()
		}

		for lane := 4; lane < 8; lane++ {
			acc[1] += values[index+lane].Float32()
		}

		index += 8
	}

	sumF32 := (acc[0] + acc[1]) + (acc[2] + acc[3])

	for index < count {
		sumF32 = metalF16SumScalarLaneF32Sim(sumF32, values[index])
		index++
	}

	return dtype.Fromfloat32(sumF32)
}

func metalF16SumScalarLaneF32Sim(sumF32 float32, value dtype.F16) float32 {
	sumBits := math.Float32bits(sumF32)
	sumBits = (sumBits & 0xFFFF0000) | uint32(uint16(value))
	sumF32 = math.Float32frombits(sumBits)

	return sumF32 + value.Float32()
}

func TestGroupNormF16StatsProbeSpatial1024Lane3768(t *testing.T) {
	probeGroupNormF16Lane(t, 1024, 3768)
}

func TestGroupNormF16StatsProbeBatch1Spatial8192Lane8709(t *testing.T) {
	probeGroupNormF16LaneWithConfig(t, 1, 4, 2, 8192, 8709)
}

func probeGroupNormF16LaneWithConfig(
	t *testing.T,
	batch int,
	channels int,
	groups int,
	spatial int,
	laneIndex int,
) {
	t.Helper()

	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(batch*channels*spatial, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)

	inputF16 := make([]dtype.F16, len(input))
	scaleF16 := make([]dtype.F16, len(scale))
	biasF16 := make([]dtype.F16, len(bias))

	for index := range input {
		inputF16[index] = dtype.Fromfloat32(input[index])
	}

	for index := range scale {
		scaleF16[index] = dtype.Fromfloat32(scale[index])
	}

	for index := range bias {
		biasF16[index] = dtype.Fromfloat32(bias[index])
	}

	batchIndex := laneIndex / (channels * spatial)
	channel := (laneIndex % (channels * spatial)) / spatial
	groupIndex := channel / channelsPerGroup
	groupStart := batchIndex*channels*spatial + groupIndex*channelsPerGroup*spatial
	groupSlice := inputF16[groupStart : groupStart+groupSize]
	statsRow := batchIndex*groups + groupIndex

	cpuSum := reduction.SumFloat16Native(groupSlice)
	cpuMean := cpuSum.Float32() / float32(groupSize)
	cpuVariance := groupNormVarianceF16Sim(groupSlice, cpuMean)
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))

	channelInGroup := (laneIndex - groupStart) / spatial
	scaleChannel := groupIndex*channelsPerGroup + channelInGroup
	cpuOut := (inputF16[laneIndex].Float32()-cpuMean)*cpuInv*scaleF16[scaleChannel].Float32() + biasF16[scaleChannel].Float32()

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float16)
	statsTensor := harness.UploadVector(make([]float32, batch*groups*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchGroupNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)
	metalMean := stats[statsRow*2]
	metalInv := stats[statsRow*2+1]

	scaleTensor := harness.UploadVector(scale, dtype.Float16)
	biasTensor := harness.UploadVector(bias, dtype.Float16)
	splitOutputTensor := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer splitOutputTensor.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		splitOutputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.Float16)
	splitGot := harness.DownloadFloat32(splitOutputTensor, dtype.Float16)
	splitBytes := splitOutputTensor.ReadBytes()

	t.Logf("batch=%d channels=%d spatial=%d lane=%d statsRow=%d", batch, channels, spatial, laneIndex, statsRow)
	t.Logf("cpuMean=0x%08x metalMean=0x%08x meanUlp=%d",
		math.Float32bits(cpuMean), math.Float32bits(metalMean), parity.Float32ULPDistance(cpuMean, metalMean))
	t.Logf("cpuInv=0x%08x metalInv=0x%08x invUlp=%d",
		math.Float32bits(cpuInv), math.Float32bits(metalInv), parity.Float32ULPDistance(cpuInv, metalInv))
	t.Logf("cpuOutBits=0x%04x refBits=0x%04x splitBits=0x%04x splitUlp=%d",
		dtype.Fromfloat32(cpuOut).Bits(), dtype.Fromfloat32(want[laneIndex]).Bits(),
		dtype.Frombits(uint16(splitBytes[laneIndex*2])|uint16(splitBytes[laneIndex*2+1])<<8).Bits(),
		parity.Float32ULPDistance(splitGot[laneIndex], want[laneIndex]))
}

func TestGroupNormF16StatsProbeSpatial8192Lane50303(t *testing.T) {
	probeGroupNormF16Lane(t, 8192, 50303)
}

func probeGroupNormF16Lane(t *testing.T, spatial int, laneIndex int) {
	t.Helper()

	batch := 2
	channels := 8
	groups := 2
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(batch*channels*spatial, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)

	inputF16 := make([]dtype.F16, len(input))
	scaleF16 := make([]dtype.F16, len(scale))
	biasF16 := make([]dtype.F16, len(bias))

	for index := range input {
		inputF16[index] = dtype.Fromfloat32(input[index])
	}

	for index := range scale {
		scaleF16[index] = dtype.Fromfloat32(scale[index])
	}

	for index := range bias {
		biasF16[index] = dtype.Fromfloat32(bias[index])
	}

	batchIndex := laneIndex / (channels * spatial)
	channel := (laneIndex % (channels * spatial)) / spatial
	groupIndex := channel / channelsPerGroup
	groupStart := batchIndex*channels*spatial + groupIndex*channelsPerGroup*spatial
	groupSlice := inputF16[groupStart : groupStart+groupSize]
	statsRow := batchIndex*groups + groupIndex

	cpuSum := reduction.SumFloat16Native(groupSlice)
	cpuMean := cpuSum.Float32() / float32(groupSize)
	cpuVariance := groupNormVarianceF16Sim(groupSlice, cpuMean)
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))

	channelInGroup := (laneIndex - groupStart) / spatial
	scaleChannel := groupIndex*channelsPerGroup + channelInGroup
	cpuOut := (inputF16[laneIndex].Float32()-cpuMean)*cpuInv*scaleF16[scaleChannel].Float32() + biasF16[scaleChannel].Float32()

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float16)
	statsTensor := harness.UploadVector(make([]float32, batch*groups*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchGroupNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)
	metalMean := stats[statsRow*2]
	metalInv := stats[statsRow*2+1]

	scaleTensor := harness.UploadVector(scale, dtype.Float16)
	biasTensor := harness.UploadVector(bias, dtype.Float16)
	splitOutputTensor := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer splitOutputTensor.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		splitOutputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.Float16)
	splitGot := harness.DownloadFloat32(splitOutputTensor, dtype.Float16)
	splitBytes := splitOutputTensor.ReadBytes()

	cpuStatsFull := make([]float32, batch*groups*2)
	cpuStatsFull[statsRow*2] = cpuMean
	cpuStatsFull[statsRow*2+1] = cpuInv
	cpuStatsTensorFull := harness.UploadVector(cpuStatsFull, dtype.Float32)
	cpuApplyOutput := harness.UploadVector(make([]float32, len(input)), dtype.Float16)
	defer cpuStatsTensorFull.Close()
	defer cpuApplyOutput.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		cpuApplyOutput.Ref(),
		cpuStatsTensorFull.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch cpu-stats apply: %v", err)
	}

	cpuStatsGot := harness.DownloadFloat32(cpuApplyOutput, dtype.Float16)
	cpuStatsBytes := cpuApplyOutput.ReadBytes()
	inputBytes := inputTensor.ReadBytes()
	scaleBytes := scaleTensor.ReadBytes()
	biasBytes := biasTensor.ReadBytes()
	deviceInput := dtype.Frombits(uint16(inputBytes[laneIndex*2]) | uint16(inputBytes[laneIndex*2+1])<<8)
	deviceScale := dtype.Frombits(uint16(scaleBytes[scaleChannel*2]) | uint16(scaleBytes[scaleChannel*2+1])<<8)
	deviceBias := dtype.Frombits(uint16(biasBytes[scaleChannel*2]) | uint16(biasBytes[scaleChannel*2+1])<<8)
	deviceManual := (deviceInput.Float32()-cpuMean)*cpuInv*deviceScale.Float32() + deviceBias.Float32()

	t.Logf("spatial=%d lane=%d statsRow=%d groupStart=%d", spatial, laneIndex, statsRow, groupStart)
	t.Logf("cpuSum=0x%04x cpuMean=0x%08x metalMean=0x%08x meanUlp=%d",
		uint16(cpuSum), math.Float32bits(cpuMean), math.Float32bits(metalMean), parity.Float32ULPDistance(cpuMean, metalMean))
	t.Logf("cpuInv=0x%08x metalInv=0x%08x invUlp=%d",
		math.Float32bits(cpuInv), math.Float32bits(metalInv), parity.Float32ULPDistance(cpuInv, metalInv))
	t.Logf("cpuOut=%v refOut=%v cpuOutBits=0x%04x refBits=0x%04x deviceManualBits=0x%04x",
		cpuOut, want[laneIndex], dtype.Fromfloat32(cpuOut).Bits(), dtype.Fromfloat32(want[laneIndex]).Bits(),
		dtype.Fromfloat32(deviceManual).Bits())
	t.Logf("cpuStatsApply bits=0x%04x ulp=%d metalStatsApply bits=0x%04x ulp=%d",
		dtype.Frombits(uint16(cpuStatsBytes[laneIndex*2])|uint16(cpuStatsBytes[laneIndex*2+1])<<8).Bits(),
		parity.Float32ULPDistance(cpuStatsGot[laneIndex], want[laneIndex]),
		dtype.Frombits(uint16(splitBytes[laneIndex*2])|uint16(splitBytes[laneIndex*2+1])<<8).Bits(),
		parity.Float32ULPDistance(splitGot[laneIndex], want[laneIndex]))
}

func TestGroupNormBF16StatsProbeSpatial7Lane1(t *testing.T) {
	batch := 2
	channels := 8
	groups := 2
	spatial := 7
	laneIndex := 1
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(batch*channels*spatial, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)

	inputBF16 := make([]dtype.BF16, len(input))
	scaleBF16 := make([]dtype.BF16, len(scale))
	biasBF16 := make([]dtype.BF16, len(bias))

	for index := range input {
		inputBF16[index] = dtype.NewBfloat16FromFloat32(input[index])
	}

	for index := range scale {
		scaleBF16[index] = dtype.NewBfloat16FromFloat32(scale[index])
	}

	for index := range bias {
		biasBF16[index] = dtype.NewBfloat16FromFloat32(bias[index])
	}

	batchIndex := laneIndex / (channels * spatial)
	channel := (laneIndex % (channels * spatial)) / spatial
	groupIndex := channel / channelsPerGroup
	groupStart := batchIndex*channels*spatial + groupIndex*channelsPerGroup*spatial
	groupSlice := inputBF16[groupStart : groupStart+groupSize]
	statsRow := batchIndex*groups + groupIndex

	cpuSum := reduction.SumBFloat16Native(groupSlice)
	cpuMean := (&cpuSum).Float32() / float32(groupSize)
	cpuVariance := groupNormVarianceBF16Sim(groupSlice, cpuMean)
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.BFloat16)
	statsTensor := harness.UploadVector(make([]float32, batch*groups*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchGroupNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.BFloat16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)
	metalMean := stats[statsRow*2]
	metalInv := stats[statsRow*2+1]
	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.BFloat16)

	t.Logf("statsRow=%d cpuMean=0x%08x metalMean=0x%08x meanUlp=%d",
		statsRow, math.Float32bits(cpuMean), math.Float32bits(metalMean), parity.Float32ULPDistance(cpuMean, metalMean))
	t.Logf("cpuInv=0x%08x metalInv=0x%08x invUlp=%d",
		math.Float32bits(cpuInv), math.Float32bits(metalInv), parity.Float32ULPDistance(cpuInv, metalInv))
	t.Logf("want[%d]=%v", laneIndex, want[laneIndex])

	scaleTensor := harness.UploadVector(scale, dtype.BFloat16)
	biasTensor := harness.UploadVector(bias, dtype.BFloat16)
	outputTensor := harness.UploadVector(make([]float32, len(input)), dtype.BFloat16)
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	cpuStatsFull := make([]float32, batch*groups*2)
	cpuStatsFull[statsRow*2] = cpuMean
	cpuStatsFull[statsRow*2+1] = cpuInv
	cpuStatsTensor := harness.UploadVector(cpuStatsFull, dtype.Float32)
	defer cpuStatsTensor.Close()

	if err := DispatchGroupNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		cpuStatsTensor.Ref(),
		dtype.BFloat16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.BFloat16)
	gotBytes := outputTensor.ReadBytes()
	t.Logf("cpuStatsApply bits=0x%04x ulp=%d",
		uint16(gotBytes[laneIndex*2])|uint16(gotBytes[laneIndex*2+1])<<8,
		parity.Float32ULPDistance(got[laneIndex], want[laneIndex]))
}

func groupNormVarianceBF16Sim(groupSlice []dtype.BF16, mean float32) float32 {
	var variance float32

	for index := range groupSlice {
		delta := (&groupSlice[index]).Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(groupSlice))
}

func TestGroupNormF16SumSimulationSpatial1024(t *testing.T) {
	batch := 2
	channels := 8
	groups := 2
	spatial := 1024
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(batch*channels*spatial, seedBase)
	inputF16 := make([]dtype.F16, len(input))

	for index := range input {
		inputF16[index] = dtype.Fromfloat32(input[index])
	}

	groupSlice := inputF16[:groupSize]
	cpuSum := reduction.SumFloat16Native(groupSlice)
	metalSum := simulateMetalF16SumNative(groupSlice)

	if uint16(cpuSum) != uint16(metalSum) {
		t.Fatalf("sum mismatch cpu=0x%04x metal=0x%04x", uint16(cpuSum), uint16(metalSum))
	}
}
