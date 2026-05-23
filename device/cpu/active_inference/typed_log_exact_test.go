//go:build arm64

package active_inference

import (
	"fmt"
	"math"
	"testing"
)

func TestActiveInferenceLogExactParity(t *testing.T) {
	for _, value := range []float64{activeInferenceEps, 0.05, 0.06, 0.07, 1.0, 2.5} {
		got := activeInferenceLogExact(value)
		want := math.Log(value)

		if got != want {
			t.Fatalf("value=%g got=%g want=%g", value, got, want)
		}
	}
}

func TestActiveInferenceLogF64NEONAsmParity(t *testing.T) {
	values := []float64{0.05, 0.06, 0.07, 1.0, 2.5, activeInferenceEps}

	for _, value := range values {
		t.Run(fmt.Sprintf("v=%g", value), func(testing *testing.T) {
			got := activeInferenceLogF64NEONAsm(value)
			want := activeInferenceLogExact(value)

			if got != want {
				testing.Fatalf("value=%g got=%g want=%g", value, got, want)
			}
		})
	}
}
