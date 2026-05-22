//go:build arm64

package masking

import "testing"

func TestCausalArgProbe(t *testing.T) {
	got := make([]float32, 4)
	CausalMaskArgProbeAsm(&got[0], 2, 2)
	t.Logf("probe=%v", got)
}
