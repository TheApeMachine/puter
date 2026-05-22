//go:build amd64

package causal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const causalAVX2SSE2MaxULP = 0

func avx2CausalAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2CausalAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestCateF32AVX2Parity(t *testing.T) {
	if !avx2CausalAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given CateFloat32AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match cateF32Generic for N=%d", length), func() {
				treated := randomCausalFloat32Slice(length, 0xCA40+int64(length))
				control := randomCausalFloat32Slice(length, 0xCA41+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				cateF32Generic(treated, control, want)
				CateFloat32AVX2Asm(&treated[0], &control[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX2SSE2MaxULP)
			})
		}
	})
}

func TestCateF32SSE2Parity(t *testing.T) {
	if !sse2CausalAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given CateFloat32SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match cateF32Generic for N=%d", length), func() {
				treated := randomCausalFloat32Slice(length, 0xCA42+int64(length))
				control := randomCausalFloat32Slice(length, 0xCA43+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				cateF32Generic(treated, control, want)
				CateFloat32SSE2Asm(&treated[0], &control[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX2SSE2MaxULP)
			})
		}
	})
}

func TestCounterfactualF32AVX2Parity(t *testing.T) {
	if !avx2CausalAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given CounterfactualFloat32AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match counterfactualF32Generic for N=%d", length), func() {
				observedY := randomCausalFloat32Slice(length, 0xCA44+int64(length))
				observedX := randomCausalFloat32Slice(length, 0xCA45+int64(length))
				counterfactualX := randomCausalFloat32Slice(length, 0xCA46+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				const slope = float32(-1.25)

				counterfactualF32Generic(want, observedY, observedX, counterfactualX, slope)
				CounterfactualFloat32AVX2Asm(
					&got[0], &observedY[0], &observedX[0], &counterfactualX[0],
					slope, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX2SSE2MaxULP)
			})
		}
	})
}

func TestCounterfactualF32SSE2Parity(t *testing.T) {
	if !sse2CausalAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given CounterfactualFloat32SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match counterfactualF32Generic for N=%d", length), func() {
				observedY := randomCausalFloat32Slice(length, 0xCA47+int64(length))
				observedX := randomCausalFloat32Slice(length, 0xCA48+int64(length))
				counterfactualX := randomCausalFloat32Slice(length, 0xCA49+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)
				const slope = float32(-1.25)

				counterfactualF32Generic(want, observedY, observedX, counterfactualX, slope)
				CounterfactualFloat32SSE2Asm(
					&got[0], &observedY[0], &observedX[0], &counterfactualX[0],
					slope, length,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, causalAVX2SSE2MaxULP)
			})
		}
	})
}

func TestStridedDotF32AVX2Parity(t *testing.T) {
	if !avx2CausalAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given StridedDotFloat32AVX2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match stridedDotF32Generic for N=%d", length), func() {
				const stride = 7
				values := randomCausalFloat32Slice(length*stride, 0xCA4A+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA4B+int64(length))

				want := stridedDotF32Generic(values, stride, weights, length)
				got := StridedDotFloat32AVX2Asm(&values[0], stride, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			})
		}

		convey.Convey("It should match stridedDotF32Generic at stride 1", func() {
			for _, length := range parity.Lengths {
				values := randomCausalFloat32Slice(length, 0xCA4C+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA4D+int64(length))

				want := stridedDotF32Generic(values, 1, weights, length)
				got := StridedDotFloat32AVX2Asm(&values[0], 1, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			}
		})
	})
}

func TestStridedDotF32SSE2Parity(t *testing.T) {
	if !sse2CausalAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given StridedDotFloat32SSE2Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match stridedDotF32Generic for N=%d", length), func() {
				const stride = 7
				values := randomCausalFloat32Slice(length*stride, 0xCA4E+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA4F+int64(length))

				want := stridedDotF32Generic(values, stride, weights, length)
				got := StridedDotFloat32SSE2Asm(&values[0], stride, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			})
		}

		convey.Convey("It should match stridedDotF32Generic at stride 1", func() {
			for _, length := range parity.Lengths {
				values := randomCausalFloat32Slice(length, 0xCA50+int64(length))
				weights := randomCausalFloat32Slice(length, 0xCA51+int64(length))

				want := stridedDotF32Generic(values, 1, weights, length)
				got := StridedDotFloat32SSE2Asm(&values[0], 1, &weights[0], length)

				assertStridedDotF32Parity(t, got, want, length)
			}
		})
	})
}
