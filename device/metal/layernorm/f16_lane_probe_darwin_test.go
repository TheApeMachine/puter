//go:build darwin && cgo

package layernorm

import (
	"math"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormF16Lane306Probe(t *testing.T) {
	rows := 7
	cols := 8192
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)
	want := metalparity.LayerNormReference(input, scale, bias, rows, cols, dtype.Float16)

	inputF16 := make([]dtype.F16, elementCount)
	scaleF16 := make([]dtype.F16, cols)
	biasF16 := make([]dtype.F16, cols)
	cpuOutF16 := make([]dtype.F16, elementCount)

	for index := range input {
		inputF16[index] = dtype.Fromfloat32(input[index])
	}

	for index := range scale {
		scaleF16[index] = dtype.Fromfloat32(scale[index])
	}

	for index := range bias {
		biasF16[index] = dtype.Fromfloat32(bias[index])
	}

	cpulayernorm.New().LayerNorm(
		unsafe.Pointer(&inputF16[0]),
		unsafe.Pointer(&scaleF16[0]),
		unsafe.Pointer(&biasF16[0]),
		unsafe.Pointer(&cpuOutF16[0]),
		rows,
		cols,
		dtype.Float16,
	)

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float16)
	scaleTensor := harness.UploadVector(scale, dtype.Float16)
	biasTensor := harness.UploadVector(bias, dtype.Float16)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float16)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	if err := DispatchLayerNormRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		dtype.Float16,
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float16)

	laneIndex := 306
	rowIndex := laneIndex / cols
	colIndex := laneIndex % cols
	rowF16 := inputF16[rowIndex*cols : (rowIndex+1)*cols]

	var mean float32
	for index := range rowF16 {
		mean += rowF16[index].Float32()
	}
	mean /= float32(len(rowF16))

	var variance float32
	for index := range rowF16 {
		delta := rowF16[index].Float32() - mean
		variance += delta * delta
	}
	variance /= float32(len(rowF16))
	invStdDev := float32(1.0 / math.Sqrt(float64(variance+1e-5)))

	normalized := (rowF16[colIndex].Float32() - mean) * invStdDev
	resultF32 := normalized*scaleF16[colIndex].Float32() + biasF16[colIndex].Float32()
	manualF16 := dtype.Fromfloat32(resultF32)
	cpuMean := float32(0)
	cpuVariance := float32(0)

	for index := range rowF16 {
		cpuMean += rowF16[index].Float32()
	}

	cpuMean /= float32(len(rowF16))

	for index := range rowF16 {
		delta := rowF16[index].Float32() - cpuMean
		cpuVariance += delta * delta
	}

	cpuVariance /= float32(len(rowF16))
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))
	cpuNorm := (rowF16[colIndex].Float32() - cpuMean) * cpuInv
	cpuManual := cpuNorm*scaleF16[colIndex].Float32() + biasF16[colIndex].Float32()
	effectiveInv := ((got[laneIndex] - biasF16[colIndex].Float32()) / scaleF16[colIndex].Float32()) / (rowF16[colIndex].Float32() - cpuMean)

	t.Logf(
		"lane %d mean=%.9g cpuMean=%.9g inv=%.9g cpuInv=%.9g effectiveInv=%.9g resultF32=%.9g cpuManual=%.9g manualF16Bits=%#x want=%.9g cpuDirect=%.9g got=%.9g wantBits=%#x gotBits=%#x gotUlp=%d manualVsWant=%d cpuManualVsDirect=%d",
		laneIndex,
		mean, cpuMean, invStdDev, cpuInv, effectiveInv, resultF32, cpuManual,
		uint16(manualF16),
		want[laneIndex], cpuOutF16[laneIndex].Float32(), got[laneIndex],
		math.Float32bits(want[laneIndex]),
		math.Float32bits(got[laneIndex]),
		cpuparity.Float32ULPDistance(got[laneIndex], want[laneIndex]),
		cpuparity.Float32ULPDistance(resultF32, want[laneIndex]),
		cpuparity.Float32ULPDistance(cpuManual, cpuOutF16[laneIndex].Float32()),
	)
}
