//go:build arm64

package sampling

import (
	"fmt"
	"testing"

	cpumath "github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestSamExpDebugN1(t *testing.T) {
	logits := []float32{2.5}
	want := make([]float32, 1)
	got := make([]float32, 1)
	SamplingSoftmaxRowGeneric(logits, want, 1.0)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 1.0, 1)
	fmt.Printf("N=1 logits=%v want=%g got=%g exp0=%g\n", logits, want[0], got[0], cpumath.FastExp32(0))

	cases := []float32{0, -18.256193, -25.682960510253906, 2.5, -87.33654, 88.0}

	for _, value := range cases {
		wantExp := cpumath.FastExp32(value)
		gotExp := samFastExp32OneNEONAsm(value)
		ulp := parity.Float32ULPDistance(gotExp, wantExp)

		fmt.Printf("fastExp x=%g want=%g got=%g ulp=%d\n", value, wantExp, gotExp, ulp)

		if ulp > 2 {
			t.Fatalf("fastExp x=%g ulp=%d max=2", value, ulp)
		}
	}
}

func TestSamExpWorstLaneN64(t *testing.T) {
	logits := randomSamplingLogits(64, 0x3610+64)
	maximum := logits[0]

	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}

	shifted := (logits[34] - maximum) / 0.85
	want := cpumath.FastExp32(shifted)
	got := samFastExp32OneNEONAsm(shifted)
	ulp := parity.Float32ULPDistance(got, want)

	fmt.Printf(
		"lane34 shifted=%.17g want=%.17g got=%.17g ulp=%d\n",
		shifted, want, got, ulp,
	)

	if ulp > 2 {
		t.Fatalf("exp ulp=%d max=2", ulp)
	}
}
