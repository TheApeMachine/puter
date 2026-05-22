//go:build amd64

package layernorm

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const layernormReducedMaxULP = 0

func avx2LayerNormAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2LayerNormAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestLayerNormSquaredDiffSumF32AVX2Parity(t *testing.T) {
	if !avx2LayerNormAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given LayerNormSquaredDiffSum float32 AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, _, _ := randomLayerNormRow(length, 0x5A00+int64(length))
				mean := float32(0.125)

				want := LayerNormSquaredDiffSumGeneric(row, mean)
				got := layerNormSquaredDiffSumF32AVX2(row, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			})
		}
	})
}

func TestLayerNormApplyRowF32AVX2Parity(t *testing.T) {
	if !avx2LayerNormAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given LayerNormApplyRow float32 AVX2", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, scale, bias := randomLayerNormRow(length, 0x5A02+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				layerNormApplyRowF32AVX2(got, row, scale, bias, mean, invStdDev)

				parity.AssertFloat32SlicesWithinULP(t, got, want, layernormReducedMaxULP)
			})
		}
	})
}

func TestLayerNormSquaredDiffSumF32SSE2Parity(t *testing.T) {
	if !sse2LayerNormAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given LayerNormSquaredDiffSum float32 SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, _, _ := randomLayerNormRow(length, 0x5B00+int64(length))
				mean := float32(0.125)

				want := LayerNormSquaredDiffSumGeneric(row, mean)
				got := layerNormSquaredDiffSumF32SSE2(row, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			})
		}
	})
}

func TestLayerNormApplyRowF32SSE2Parity(t *testing.T) {
	if !sse2LayerNormAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given LayerNormApplyRow float32 SSE2", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row, scale, bias := randomLayerNormRow(length, 0x5B02+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				LayerNormApplyRowGeneric(want, row, scale, bias, mean, invStdDev)
				layerNormApplyRowF32SSE2(got, row, scale, bias, mean, invStdDev)

				parity.AssertFloat32SlicesWithinULP(t, got, want, layernormReducedMaxULP)
			})
		}
	})
}
