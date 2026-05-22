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
