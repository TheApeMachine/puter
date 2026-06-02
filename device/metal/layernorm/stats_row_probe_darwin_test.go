//go:build darwin && cgo

package layernorm

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/cpu/parity"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormStatsRow1Probe(t *testing.T) {
	rows := 4
	cols := 64
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float32)
	statsTensor := harness.UploadVector(make([]float32, rows*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchLayerNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
		dtype.Float32,
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	stats := harness.DownloadFloat32(statsTensor, dtype.Float32)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*cols : (rowIndex+1)*cols]
		sum := cpulayernorm.SumFloat32Native(row)
		wantMean := float32(float64(sum) / float64(cols))
		variance := float64(cpulayernorm.LayerNormSquaredDiffSumNative(row, wantMean)) / float64(cols)
		wantInvStdDev := float32(1.0 / math.Sqrt(variance+1e-5))

		gotMean := stats[rowIndex*2]
		gotInvStdDev := stats[rowIndex*2+1]

		t.Logf(
			"row %d mean got=%.9g want=%.9g meanUlp=%d inv got=%.9g want=%.9g invUlp=%d",
			rowIndex,
			gotMean, wantMean, parity.Float32ULPDistance(gotMean, wantMean),
			gotInvStdDev, wantInvStdDev, parity.Float32ULPDistance(gotInvStdDev, wantInvStdDev),
		)
	}

	rowIndex := 1
	colIndex := 10
	laneIndex := rowIndex*cols + colIndex
	row := input[rowIndex*cols : (rowIndex+1)*cols]
	sum := cpulayernorm.SumFloat32Native(row)
	cpuMean := float32(float64(sum) / float64(cols))
	variance := float64(cpulayernorm.LayerNormSquaredDiffSumNative(row, cpuMean)) / float64(cols)
	cpuInvStdDev := float32(1.0 / math.Sqrt(variance+1e-5))

	metalMean := stats[rowIndex*2]
	metalInvStdDev := stats[rowIndex*2+1]

	withCPUStats := (input[laneIndex]-cpuMean)*cpuInvStdDev*scale[colIndex] + bias[colIndex]
	withMetalStats := (input[laneIndex]-metalMean)*metalInvStdDev*scale[colIndex] + bias[colIndex]
	withCPUStatsFMA := scale[colIndex]*(input[laneIndex]-cpuMean)*cpuInvStdDev + bias[colIndex]
	withMetalStatsFMA := scale[colIndex]*(input[laneIndex]-metalMean)*metalInvStdDev + bias[colIndex]

	t.Logf(
		"lane %d cpuStats=%.9g metalStats=%.9g cpuFMA=%.9g metalFMA=%.9g",
		laneIndex,
		withCPUStats, withMetalStats, withCPUStatsFMA, withMetalStatsFMA,
	)
}
