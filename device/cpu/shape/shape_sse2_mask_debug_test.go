//go:build amd64

package shape

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestMaskedFillFloat32SSE2AsmMaskDebug(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 required")
	}

	input := []float32{1, 2, 3, 4}
	mask := []byte{0x02}
	fillValue := float32(-0.75)
	got := make([]float32, 4)
	want := []float32{1, -0.75, 3, 4}

	MaskedFillGeneric(want, input, fillValue, mask)
	MaskedFillFloat32SSE2Asm(&got[0], &input[0], fillValue, &mask[0], 4)

	for index := range got {
		if got[index] != want[index] {
			t.Fatalf("lane %d got=%v want=%v", index, got[index], want[index])
		}
	}
}

func TestWhereFloat32SSE2AsmMaskDebug(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 required")
	}

	positive := []float32{1, 2, 3, 4, 5, 6, 7}
	negative := []float32{-1, -2, -3, -4, -5, -6, -7}
	mask := []byte{0x0A} // bits 1 and 3 set
	got := make([]float32, 7)
	want := []float32{-1, 2, -3, 4, -5, -6, -7}

	WhereGeneric(want, positive, negative, mask)
	WhereFloat32SSE2Asm(&got[0], &positive[0], &negative[0], &mask[0], 7)

	for index := range got {
		if got[index] != want[index] {
			t.Fatalf("lane %d got=%v want=%v", index, got[index], want[index])
		}
	}
}
