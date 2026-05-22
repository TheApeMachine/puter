//go:build arm64

package masking

import (
	"testing"
	"unsafe"
)

func TestCausalMaskNEONDebug(t *testing.T) {
	got := make([]float32, 4)
	for index := range got {
		got[index] = 999
	}
	want := make([]float32, 4)
	causalMaskF32Generic(unsafe.Pointer(&want[0]), 2, 2)

	CausalMaskArgProbeAsm(&got[0], 2, 2)
	t.Logf("after probe=%v", got)

	got = make([]float32, 4)
	for index := range got {
		got[index] = 999
	}
	CausalMaskFloat32NEONAsm(&got[0], 2, 2)
	t.Logf("generic=%v", want)
	t.Logf("neon=%v", got)
	for index, value := range got {
		if value != want[index] {
			t.Fatalf("index %d got=%v want=%v", index, value, want[index])
		}
	}
}

func TestALiBiNEONDebug(t *testing.T) {
	scores := []float32{0.30508983}
	slope := []float32{0.25}
	got := make([]float32, 1)
	want := make([]float32, 1)
	alibiBiasF32Generic(
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&slope[0]),
		unsafe.Pointer(&want[0]),
		1,
		1,
	)
	ALiBiBiasF32NEON(&scores[0], &slope[0], &got[0], 1, 1)
	t.Logf("want=%v got=%v", want[0], got[0])
	if got[0] != want[0] {
		t.Fatalf("got=%v want=%v", got[0], want[0])
	}
}
