//go:build arm64

package masking

import (
	"math"
	"testing"
)

func TestCausalMaskNEONDebug(t *testing.T) {
	got := make([]float32, 4)
	CausalMaskFloat32NEONAsm(&got[0], 2, 2)
	t.Logf("causal got=%v", got)
	for index, value := range got {
		if index == 1 && value != float32(math.Inf(-1)) {
			t.Fatalf("index 1 got=%v want -Inf", value)
		}
	}
}

func TestALiBiNEONDebug(t *testing.T) {
	scores := []float32{0.30508983}
	slope := []float32{0.25}
	got := make([]float32, 1)
	ALiBiBiasFloat32NEONAsm(&scores[0], &slope[0], &got[0], 1, 1)
	t.Logf("alibi got=%v want=%v", got[0], scores[0])
	if got[0] != scores[0] {
		t.Fatalf("got=%v want=%v", got[0], scores[0])
	}
}
