//go:build arm64

package active_inference

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestFreeEnergyBF16NEONDebug(t *testing.T) {
	likelihood := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.05)}
	posterior := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.07)}
	prior := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.06)}
	want := FreeEnergyBFloat16Scalar(likelihood, posterior, prior)
	got := FreeEnergyBF16NEON(likelihood, posterior, prior)

	like := float64(likelihood[0].Float32())
	post := float64(posterior[0].Float32())
	pri := float64(prior[0].Float32())
	cl := math.Max(activeInferenceEps, like)
	cp := math.Max(activeInferenceEps, post)
	cpr := math.Max(activeInferenceEps, pri)
	sum := -post*math.Log(cl) + post*(math.Log(cp)-math.Log(cpr))

	t.Logf("want=%v (%d) f32=%v got=%v (%d) f32=%v manual=%v",
		want, uint16(want), want.Float32(), got, uint16(got), got.Float32(), sum)
	t.Logf("like=%v post=%v pri=%v log(cl)=%v", like, post, pri, math.Log(cl))
	t.Logf("bridge=%v", activeInferenceStdLogF64(cl))
}
