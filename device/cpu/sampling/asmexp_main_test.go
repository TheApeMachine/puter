//go:build arm64

package sampling

import (
	"fmt"
	"testing"

	cpumath "github.com/theapemachine/puter/device/cpu/math"
)

func TestAsmExpDirect(t *testing.T) {
	for _, value := range []float32{0, -18.256193, 2.5} {
		want := cpumath.FastExp32(value)
		got := samFastExp32OneNEONAsm(value)
		fmt.Printf("direct x=%g want=%g got=%g\n", value, want, got)
	}
}
