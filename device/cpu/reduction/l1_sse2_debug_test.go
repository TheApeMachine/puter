//go:build amd64

package reduction

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestL1NormBF16SSE2Debug(t *testing.T) {
	if !bf16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for bf16 widen")
	}
	for _, length := range []int{1, 7, 64} {
		values := make([]uint16, length)

		for index := range values {
			values[index] = uint16(dtype.NewBfloat16FromFloat32(float32(index + 1)))
		}

		want := L1NormBF16Generic(&values[0], length)
		got := L1NormBF16SSE2(&values[0], length)

		if got != want {
			t.Fatalf("N=%d got=%g want=%g", length, got, want)
		}
	}
}

func TestSumBF16SSE2Debug(t *testing.T) {
	if !bf16SSE2ReductionAvailable() {
		t.Skip("SSE2+AVX required for bf16 widen")
	}
	for _, length := range []int{1, 7, 64} {
		values := make([]uint16, length)

		for index := range values {
			values[index] = uint16(dtype.NewBfloat16FromFloat32(float32(index + 1)))
		}

		wantBits := SumBF16Generic(&values[0], length)
		gotBits := SumBF16SSE2(&values[0], length)

		if gotBits != wantBits {
			t.Fatalf("N=%d got=%x want=%x", length, gotBits, wantBits)
		}
	}
}
