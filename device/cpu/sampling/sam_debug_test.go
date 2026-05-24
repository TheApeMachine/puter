//go:build arm64

package sampling

import (
	"fmt"
	"testing"
)

func TestDebugN1(t *testing.T) {
	logits := randomSamplingLogits(1, 0x3610+1)
	want := make([]float32, 1)
	got := make([]float32, 1)
	SamplingSoftmaxRowGeneric(logits, want, 0.85)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 0.85, 1)
	fmt.Printf("logit=%v want=%v got=%v\n", logits[0], want[0], got[0])
}
