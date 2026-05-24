//go:build arm64

package sampling

import (
	"fmt"
	"testing"

	cpumath "github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestExpVsNormalizeN64(t *testing.T) {
	length := 64
	logits := randomSamplingLogits(length, 0x3610+int64(length))
	temperature := float32(0.85)

	maximum := logits[0]
	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}

	exps := make([]float32, length)
	var sum float64
	for index, value := range logits {
		exps[index] = cpumath.FastExp32((value - maximum) / temperature)
		sum += float64(exps[index])
	}
	scale := float32(1.0 / sum)

	want := make([]float32, length)
	for index := range exps {
		want[index] = exps[index] * scale
	}

	neonOut := make([]float32, length)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &neonOut[0], temperature, length)

	maxExpULP := 0
	maxNormULP := 0
	for index := range exps {
		// recover neon exp assuming same scale as generic
		neonExp := neonOut[index] * float32(sum)
		expULP := parity.Float32ULPDistance(neonExp, exps[index])
		if expULP > maxExpULP {
			maxExpULP = expULP
		}
		normULP := parity.Float32ULPDistance(neonOut[index], want[index])
		if normULP > maxNormULP {
			maxNormULP = normULP
		}
	}
	fmt.Printf("maxExpULP=%d maxNormULP=%d sum=%g\n", maxExpULP, maxNormULP, sum)
}
