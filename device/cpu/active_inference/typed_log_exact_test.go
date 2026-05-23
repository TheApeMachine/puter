//go:build arm64

package active_inference

import (
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

func TestActiveInferenceLogBridge(t *testing.T) {
	for _, value := range []float64{activeInferenceEps, 0.05, 0.07, 1.0} {
		got := activeInferenceLogF64(value)
		want := math.Log(value)

		if got != want {
			t.Fatalf("value=%g got=%g want=%g", value, got, want)
		}
	}
}
