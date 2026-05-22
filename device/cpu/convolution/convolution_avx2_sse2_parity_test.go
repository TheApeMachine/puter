//go:build amd64

package convolution

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func TestConvPatchDotF32AVX2Parity(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given ConvPatchDotF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ConvPatchDotScalar for N=%d", length), func() {
				weight, patch := randomConvolutionFloat32Pair(length, 0x2500+int64(length))
				want := ConvPatchDotScalar(weight, patch, length)
				got := ConvPatchDotF32AVX2(&weight[0], &patch[0], length)

				assertConvPatchDotParity(t, got, want, length)
			})
		}
	})
}

func TestConvPatchDotF32SSE2Parity(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given ConvPatchDotF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ConvPatchDotScalar for N=%d", length), func() {
				weight, patch := randomConvolutionFloat32Pair(length, 0x2510+int64(length))
				want := ConvPatchDotScalar(weight, patch, length)
				got := ConvPatchDotF32SSE2(&weight[0], &patch[0], length)

				assertConvPatchDotParity(t, got, want, length)
			})
		}
	})
}
