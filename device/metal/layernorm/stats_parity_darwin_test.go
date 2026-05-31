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

func TestLayerNormMetalStatsParity(t *testing.T) {
	harness := metalparity.NewHarness(t)
	defer harness.Close()

	rows := 4
	cols := 64
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	statsTensor := harness.UploadVector(make([]float32, rows*2), dtype.Float32)
	defer inputTensor.Close()
	defer statsTensor.Close()

	if err := DispatchLayerNormStatsRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		statsTensor.Ref(),
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

		if parity.Float32ULPDistance(gotMean, wantMean) > 3 {
			t.Fatalf("row %d mean got=%v want=%v ulp=%d", rowIndex, gotMean, wantMean,
				parity.Float32ULPDistance(gotMean, wantMean))
		}

		if parity.Float32ULPDistance(gotInvStdDev, wantInvStdDev) > 3 {
			t.Fatalf("row %d invStdDev got=%v want=%v ulp=%d", rowIndex, gotInvStdDev, wantInvStdDev,
				parity.Float32ULPDistance(gotInvStdDev, wantInvStdDev))
		}
	}
}

func TestLayerNormMetalApplyParity(t *testing.T) {
	harness := metalparity.NewHarness(t)
	defer harness.Close()

	rows := 4
	cols := 64
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)
	want := metalparity.LayerNormReference(input, scale, bias, rows, cols, dtype.Float32)

	inputTensor := harness.UploadVector(input, dtype.Float32)
	scaleTensor := harness.UploadVector(scale, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
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
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch stats: %v", err)
	}

	if err := DispatchLayerNormApplyRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		statsTensor.Ref(),
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch apply: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)
	metalparity.AssertFloat32SlicesWithinULP(t, got, want, 3)
}
