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
		for index := range want {
			ulp := parity.Float32ULPDistance(got[index], want[index])
			if ulp > maxULP {
				maxULP = ulp
				worst = index
			}
		}
		if worst >= 0 {
			fmt.Printf("N=%d maxULP=%d worst=%d got=%g want=%g\n", length, maxULP, worst, got[worst], want[worst])
		} else {
			fmt.Printf("N=%d maxULP=%d\n", length, maxULP)
		}
		if maxULP > 2 {
			for index := range want {
				ulp := parity.Float32ULPDistance(got[index], want[index])
				if ulp > 2 {
					fmt.Printf("  lane %d ulp=%d got=%g want=%g\n", index, ulp, got[index], want[index])
				}
			}
			t.Fatalf("N=%d maxULP=%d", length, maxULP)
		}
	}
}
