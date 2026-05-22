package metal

import (
	"math"
	"testing"
)

func TestBatchNormFormulaTwoStepMatchesGPU(t *testing.T) {
	input := math.Float32frombits(0xbe000000)
	mean := math.Float32frombits(0xbe300000)
	scale := math.Float32frombits(0x3f780000)
	bias := math.Float32frombits(0xbd400000)
	inv := math.Float32frombits(0x3f843277)
	gpu := math.Float32frombits(0x37caa400)
	want := math.Float32frombits(0x37caa380)

	oneStep := (input - mean) * inv * scale + bias
	normalized := (input - mean) * inv
	twoStep := normalized*scale + bias

	t.Logf("oneStep=%08x twoStep=%08x gpu=%08x want=%08x",
		math.Float32bits(oneStep),
		math.Float32bits(twoStep),
		math.Float32bits(gpu),
		math.Float32bits(want),
	)
	t.Logf("ULP twoStep gpu=%d oneStep want=%d",
		float32ULPDistance(twoStep, gpu),
		float32ULPDistance(oneStep, want),
	)
}
