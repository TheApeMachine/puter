//go:build darwin && cgo

package layernorm

import (
	"math"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/cpu/reduction"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormBF16Lane29627Probe(t *testing.T) {
	rows := 7
	cols := 8192
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)
	laneIndex := 29627
	rowIndex := laneIndex / cols
	colIndex := laneIndex % cols

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)
	want := metalparity.LayerNormReference(input, scale, bias, rows, cols, dtype.BFloat16)

	inputBF16 := make([]dtype.BF16, elementCount)
	scaleBF16 := make([]dtype.BF16, cols)
	biasBF16 := make([]dtype.BF16, cols)

	for index := range input {
		inputBF16[index] = dtype.NewBfloat16FromFloat32(input[index])
	}

	for index := range scale {
		scaleBF16[index] = dtype.NewBfloat16FromFloat32(scale[index])
	}

	for index := range bias {
		biasBF16[index] = dtype.NewBfloat16FromFloat32(bias[index])
	}

	row := inputBF16[rowIndex*cols : (rowIndex+1)*cols]
	sumNative := reduction.SumBFloat16Native(row)
	cpuMean := (&sumNative).Float32() / float32(cols)
	cpuVariance := layerNormVarianceBF16Probe(row, cpuMean)
	cpuInv := float32(1.0 / math.Sqrt(float64(cpuVariance+1e-5)))
	cpuOut := ((&row[colIndex]).Float32()-cpuMean)*cpuInv*(&scaleBF16[colIndex]).Float32() + (&biasBF16[colIndex]).Float32()

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.BFloat16)
	scaleTensor := harness.UploadVector(scale, dtype.BFloat16)
	biasTensor := harness.UploadVector(bias, dtype.BFloat16)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.BFloat16)
	statsTensor := harness.UploadVector(make([]float32, rows*2), dtype.Float32)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchLayerNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.BFloat16,
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)
	metalMean := stats[rowIndex*2]
	metalInv := stats[rowIndex*2+1]

	if err := DispatchLayerNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		statsTensor.Ref(),
		dtype.BFloat16,
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.BFloat16)

	t.Logf("want[%d]=%v got[%d]=%v", laneIndex, want[laneIndex], laneIndex, got[laneIndex])
	t.Logf("cpuMean=%v metalMean=%v meanUlp=%d", cpuMean, metalMean, parity.Float32ULPDistance(metalMean, cpuMean))
	t.Logf("cpuInv=%v metalInv=%v invUlp=%d", cpuInv, metalInv, parity.Float32ULPDistance(metalInv, cpuInv))
	t.Logf("cpuOut=%v manualMetalOut=%v", cpuOut,
		((&inputBF16[laneIndex]).Float32()-metalMean)*metalInv*(&scaleBF16[colIndex]).Float32()+(&biasBF16[colIndex]).Float32())

	cpuOutput := make([]dtype.BF16, elementCount)
	cpulayernorm.New().LayerNorm(
		unsafe.Pointer(&inputBF16[0]),
		unsafe.Pointer(&scaleBF16[0]),
		unsafe.Pointer(&biasBF16[0]),
		unsafe.Pointer(&cpuOutput[0]),
		rows,
		cols,
		dtype.BFloat16,
	)
}

func layerNormVarianceBF16Probe(row []dtype.BF16, mean float32) float32 {
	var variance float32

	for index := range row {
		delta := (&row[index]).Float32() - mean
		variance += delta * delta
	}

	return variance / float32(len(row))
}
