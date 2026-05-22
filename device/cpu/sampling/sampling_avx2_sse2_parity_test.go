//go:build amd64

package sampling

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2SamplingAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2SamplingAvailable() bool {
	return cpu.X86.HasSSE2
}

const samplingAVX2SSE2MaxULP = 2

func TestGreedySampleF32AVX2Parity(t *testing.T) {
	if !avx2SamplingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given GreedySampleF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match GreedySampleGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1700+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleF32AVX2(&logits[0], length)

				if got != want {
					t.Fatalf("N=%d got=%d want=%d", length, got, want)
				}
			})
		}
	})
}

func TestSamplingSoftmaxRowF32AVX2Parity(t *testing.T) {
	if !avx2SamplingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given SamplingSoftmaxRowF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SamplingSoftmaxRowGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1720+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(0.85)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowF32AVX2(&logits[0], &got[0], temperature, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match SamplingSoftmaxRowGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x1721+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(1.25)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowFloat32AVX2Asm(
					&logits[0], &got[0], temperature, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX2SSE2MaxULP)
			}
		})
	})
}

func TestGreedySampleF32SSE2Parity(t *testing.T) {
	if !sse2SamplingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given GreedySampleF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match GreedySampleGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1710+int64(length))

				want := GreedySampleGeneric(logits)
				got := GreedySampleF32SSE2(&logits[0], length)

				if got != want {
					t.Fatalf("N=%d got=%d want=%d", length, got, want)
				}
			})
		}
	})
}

func TestSamplingSoftmaxRowF32SSE2Parity(t *testing.T) {
	if !sse2SamplingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given SamplingSoftmaxRowF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match SamplingSoftmaxRowGeneric for N=%d", length), func() {
				logits := randomSamplingLogits(length, 0x1730+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(0.85)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowF32SSE2(&logits[0], &got[0], temperature, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX2SSE2MaxULP)
			})
		}

		convey.Convey("It should match SamplingSoftmaxRowGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				logits := randomSamplingLogits(length, 0x1731+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				temperature := float32(1.25)

				SamplingSoftmaxRowGeneric(logits, want, temperature)
				SamplingSoftmaxRowFloat32SSE2Asm(
					&logits[0], &got[0], temperature, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, samplingAVX2SSE2MaxULP)
			}
		})
	})
}
