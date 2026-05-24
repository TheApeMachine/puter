//go:build arm64
package sampling
import (
	"fmt"
	"testing"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
)
func TestZeroWantLanes1024(t *testing.T) {
	logits := randomSamplingLogits(1024, 0x3610+1024)
	max := logits[0]
	for _, v := range logits[1:] { if v > max { max = v } }
	want := make([]float32, 1024)
	got := make([]float32, 1024)
	SamplingSoftmaxRowGeneric(logits, want, 0.85)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 0.85, 1024)
	for i := range want {
		if want[i] == 0 && got[i] != 0 {
			input := (logits[i]-max)/0.85
			fmt.Printf("lane %d input=%g want=0 got=%g exp=%g\n", i, input, got[i], cpumath.FastExp32(input))
		}
	}
}
