//go:build arm64

package sampling

import (
	"fmt"
	"testing"

	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestScanAllLengths(t *testing.T) {
	for _, length := range parity.Lengths {
		logits := randomSamplingLogits(length, 0x3610+int64(length))
		want := make([]float32, length)
		got := make([]float32, length)
		SamplingSoftmaxRowGeneric(logits, want, 0.85)
		SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 0.85, length)
		maxULP := 0
		worst := -1
		for i := range want {
			u := parity.Float32ULPDistance(got[i], want[i])
			if u > maxULP {
				maxULP = u
				worst = i
			}
		}
		fmt.Printf("N=%d maxULP=%d worst=%d got=%g want=%g\n", length, maxULP, worst, got[worst], want[worst])
		if maxULP > 2 {
			t.Fatalf("N=%d maxULP=%d", length, maxULP)
		}
	}
}
