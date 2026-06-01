//go:build darwin && cgo

package layernorm

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormLane16Probe(t *testing.T) {
	rows := 4
	cols := 64
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)
	want := metalparity.LayerNormReference(input, scale, bias, rows, cols, dtype.Float32)

	row := input[0:cols]
	sum := cpulayernorm.SumFloat32Native(row)
	mean := float32(float64(sum) / float64(cols))
	variance := float64(cpulayernorm.LayerNormSquaredDiffSumNative(row, mean)) / float64(cols)
	invStdDev := float32(1.0 / math.Sqrt(variance+1e-5))

	outRow := make([]float32, cols)
	cpulayernorm.LayerNormApplyRowNative(outRow, row, scale, bias, mean, invStdDev)

	genericOut := make([]float32, cols)
	cpulayernorm.LayerNormApplyRowGeneric(genericOut, row, scale, bias, mean, invStdDev)

	// Isolate NEON asm block at indices 16-19 only.
	blockOut := make([]float32, 4)
	cpulayernorm.LayerNormApplyRowNEONAsm(
		&blockOut[0], &row[16], &scale[16], &bias[16],
		4, mean, invStdDev,
	)

	for _, idx := range []int{16, 17, 18, 19, 20} {
		blockValue := float32(0)
		blockUlp := -1

		if idx >= 16 && idx <= 19 {
			blockValue = blockOut[idx-16]
			blockUlp = cpuparity.Float32ULPDistance(blockValue, want[idx])
		}

		t.Logf("idx=%d manual=%.9g generic=%.9g native=%.9g block=%.9g want=%.9g manualUlp=%d genericUlp=%d nativeUlp=%d blockUlp=%d",
			idx,
			(row[idx]-mean)*invStdDev*scale[idx]+bias[idx],
			genericOut[idx], outRow[idx], blockValue, want[idx],
			cpuparity.Float32ULPDistance((row[idx]-mean)*invStdDev*scale[idx]+bias[idx], want[idx]),
			cpuparity.Float32ULPDistance(genericOut[idx], want[idx]),
			cpuparity.Float32ULPDistance(outRow[idx], want[idx]),
			blockUlp)
	}
}
