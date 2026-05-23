//go:build arm64

package active_inference

import "testing"

func TestLogIdentity(t *testing.T) {
	v := 2.5
	got := activeInferenceLogF64NEONAsm(v)
	if got != v {
		t.Fatalf("got=%g want=%g", got, v)
	}
}
