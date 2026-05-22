//go:build amd64

package rope

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const ropeReducedMaxULP = 0

func avx2RopeAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2RopeAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestRopePairsFloat32AVX2Parity(t *testing.T) {
	if !avx2RopeAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given RopePairs float32 AVX2", t, func() {
		for _, pairCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for pairs=%d", pairCount), func() {
				in, cos, sin := randomRopePairBuffers(pairCount, 0x4A00+int64(pairCount))
				want := make([]float32, 2*pairCount)
				got := make([]float32, 2*pairCount)

				RopePairsGeneric(want, in, cos, sin)
				ropePairsF32AVX2(got, in, cos, sin)

				parity.AssertFloat32SlicesWithinULP(t, got, want, ropeReducedMaxULP)
			})
		}
	})
}

func TestRopePairsFloat32SSE2Parity(t *testing.T) {
	if !sse2RopeAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given RopePairs float32 SSE2", t, func() {
		for _, pairCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for pairs=%d", pairCount), func() {
				in, cos, sin := randomRopePairBuffers(pairCount, 0x4B00+int64(pairCount))
				want := make([]float32, 2*pairCount)
				got := make([]float32, 2*pairCount)

				RopePairsGeneric(want, in, cos, sin)
				ropePairsF32SSE2(got, in, cos, sin)

				parity.AssertFloat32SlicesWithinULP(t, got, want, ropeReducedMaxULP)
			})
		}
	})
}
