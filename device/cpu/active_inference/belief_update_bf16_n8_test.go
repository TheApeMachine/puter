//go:build arm64

package active_inference

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestBeliefUpdateBF16NEONN8(t *testing.T) {
	length := 8
	likelihood := make([]dtype.BF16, length)
	prior := make([]dtype.BF16, length)
	want := make([]dtype.BF16, length)
	got := make([]dtype.BF16, length)
	for index := range likelihood {
		likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.1)
		prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.08)
	}
	BeliefUpdateBFloat16Scalar(likelihood, prior, want)
	BeliefUpdateBF16NEON(likelihood, prior, got)
	for index := range got {
		if got[index] != want[index] {
			t.Fatalf("N=8 i=%d got=%v want=%v", index, got[index], want[index])
		}
	}
}
