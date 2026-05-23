//go:build arm64

package active_inference

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestFreeEnergyBF16NEONDebugLengths(t *testing.T) {
	for _, length := range []int{1, 7, 64} {
		likelihood := make([]dtype.BF16, length)
		posterior := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%13+1) * 0.05)
			posterior[index] = dtype.NewBfloat16FromFloat32(float32(index%9+1) * 0.07)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.06)
		}

		want := FreeEnergyBF16F32LogRef(likelihood, posterior, prior)
		got := FreeEnergyBF16NEON(likelihood, posterior, prior)
		fmt.Printf("bf16 N=%d ref=%v got=%v match=%v\n", length, want, got, got == want)
		if got != want {
			t.Errorf("N=%d got=%v want=%v", length, got, want)
		}
	}
}
