//go:build arm64

package active_inference

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestFreeEnergyFP16NEONDebugLengths(t *testing.T) {
	for _, length := range []int{1, 7, 64, 1024} {
		likelihood := make([]dtype.F16, length)
		posterior := make([]dtype.F16, length)
		prior := make([]dtype.F16, length)
		for index := range likelihood {
			likelihood[index] = dtype.Fromfloat32(float32(index%13+1) * 0.05)
			posterior[index] = dtype.Fromfloat32(float32(index%9+1) * 0.07)
			prior[index] = dtype.Fromfloat32(float32(index%11+1) * 0.06)
		}

		want := FreeEnergyFP16F32LogRef(likelihood, posterior, prior)
		got := FreeEnergyFP16NEON(likelihood, posterior, prior)

		likelihoodF32 := widenF16ToF32(likelihood)
		posteriorF32 := widenF16ToF32(posterior)
		priorF32 := widenF16ToF32(prior)
		f32Direct := FreeEnergyF32NEON(&likelihoodF32[0], &posteriorF32[0], &priorF32[0], length)

		fmt.Printf("N=%d ref=%v got=%v f32=%g match=%v\n", length, want, got, f32Direct, got == want)
		if got != want {
			t.Errorf("N=%d got=%v want=%v", length, got, want)
		}
	}
}
