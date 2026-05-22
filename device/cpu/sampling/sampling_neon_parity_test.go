//go:build arm64

package sampling

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const samplingNEONMaxULP = 2

func TestGreedySampleF32NEONParity(t *testing.T) {
	convey.Convey("Given GreedySampleF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match GreedySampleGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x3600+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleF32NEON(&logits[0], length)

				if got != want {
					t.Fatalf("N=%d got=%d want=%d", length, got, want)
				}
			})
		}

		convey.Convey("It should match GreedySampleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x3601+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleFloat32NEONAsm(&logits[0], length)

				if got != want {
					t.Fatalf("direct asm N=%d got=%d want=%d", length, got, want)
				}
			})
		})
	})
}

func TestSamplingSoftmaxRowF32NEONParity(t *testing.T) {
	convey.Convey("Given SamplingSoftmaxRowF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SamplingSoftmaxRowGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x3610+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(0.85)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowF32NEON(&logits[0], &got[0], temperature, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingNEONMaxULP)
			})
		}

		convey.Convey("It should match SamplingSoftmaxRowGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x3611+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(1.25)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowFloat32NEONAsm(
					&logits[0], &got[0], temperature, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingNEONMaxULP)
			})
		})
	})
}
