//go:build darwin

package parity_test

import (
	"math"
	"math/rand"
	"testing"

	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func TestDebugSequentialF64Sum(t *testing.T) {
	rows, cols := 4, 64
	seedBase := int64(0x4F00 + rows*1000 + cols)
	rng := rand.New(rand.NewSource(seedBase))
	row := make([]float32, cols)

	for index := range row {
		row[index] = rng.Float32()*4.0 - 2.0
	}

	neonSum := reduction.SumFloat32Native(row)
	var sequentialSum float64

	for _, value := range row {
		sequentialSum += float64(value)
	}

	if math.Abs(float64(neonSum)-sequentialSum) > 1e-3 {
		t.Fatalf("neon sum %v != sequential f64 sum %v", neonSum, sequentialSum)
	}

	if math.Abs(float64(float32(sequentialSum))-float64(neonSum)) > 1e-6 {
		t.Fatalf("float32(sequential f64) %v != neon sum %v", float32(sequentialSum), neonSum)
	}

	meanF32 := float32(sequentialSum / float64(cols))
	varianceSum := cpulayernorm.LayerNormSquaredDiffSumNative(row, meanF32)
	invStdDev := float32(1.0 / math.Sqrt(float64(varianceSum)/float64(cols)+1e-5))
	scale := make([]float32, cols)
	bias := make([]float32, cols)

	for index := range scale {
		scale[index] = 1.0
	}

	out := make([]float32, cols)

	cpulayernorm.LayerNormApplyRowNative(out, row, scale, bias, meanF32, invStdDev)

	var outMean float64
	var outVariance float64

	for _, value := range out {
		outMean += float64(value)
	}

	outMean /= float64(cols)

	for _, value := range out {
		delta := float64(value) - outMean
		outVariance += delta * delta
	}

	outVariance /= float64(cols)

	if math.Abs(outMean) > 1e-4 {
		t.Fatalf("layernorm mean %v, want ~0", outMean)
	}

	if math.Abs(outVariance-1.0) > 1e-3 {
		t.Fatalf("layernorm variance %v, want ~1", outVariance)
	}
}
