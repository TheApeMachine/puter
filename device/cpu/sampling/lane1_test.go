//go:build arm64
package sampling
import (
	"fmt"
	"testing"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/cpu/parity"
)
func TestLane1N64(t *testing.T) {
	logits := randomSamplingLogits(64, 0x3610+64)
	max := logits[0]
	for _, v := range logits[1:] { if v > max { max = v } }
	want := make([]float32, 64)
	got := make([]float32, 64)
	SamplingSoftmaxRowGeneric(logits, want, 0.85)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 0.85, 64)
	i := 1
	fmt.Printf("lane1 input=%g exp=%g want=%g got=%g ulp=%d\n",
		(logits[i]-max)/0.85, cpumath.FastExp32((logits[i]-max)/0.85), want[i], got[i],
		parity.Float32ULPDistance(got[i], want[i]))
}
