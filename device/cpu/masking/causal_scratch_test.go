//go:build arm64

package masking

import (
	"math"
	"testing"
	"unsafe"
)

func TestCausalScratch2x2(t *testing.T) {
	want := make([]float32, 4)
	got := make([]float32, 4)
	for index := range got {
		got[index] = 999
	}

	causalMaskF32Generic(unsafe.Pointer(&want[0]), 2, 2)
	CausalMaskFloat32NEONAsm(&got[0], 2, 2)

	t.Logf("want=%v", want)
	t.Logf("got=%v", got)

	if got[1] != want[1] {
		t.Fatalf("index 1: got=%v want=%v", got[1], want[1])
	}

	if math.IsInf(float64(got[1]), -1) {
		t.Log("got[1] is -Inf OK")
	}
}
