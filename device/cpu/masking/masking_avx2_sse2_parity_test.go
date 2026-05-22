//go:build amd64

package masking

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2MaskingAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2MaskingAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestApplyMaskF32AVX2Parity(t *testing.T) {
	if !avx2MaskingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runApplyMaskParity(t, ApplyMaskF32AVX2, ApplyMaskFloat32AVX2Asm, 0x2820)
}

func TestApplyMaskF32SSE2Parity(t *testing.T) {
	if !sse2MaskingAvailable() {
		t.Skip("SSE2 required")
	}

	runApplyMaskParity(t, ApplyMaskF32SSE2, ApplyMaskFloat32SSE2Asm, 0x2821)
}

func TestCausalMaskF32AVX2Parity(t *testing.T) {
	if !avx2MaskingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runCausalMaskParity(t, CausalMaskF32AVX2, CausalMaskFloat32AVX2Asm)
}

func TestCausalMaskF32SSE2Parity(t *testing.T) {
	if !sse2MaskingAvailable() {
		t.Skip("SSE2 required")
	}

	runCausalMaskParity(t, CausalMaskF32SSE2, CausalMaskFloat32SSE2Asm)
}

func TestALiBiBiasF32AVX2Parity(t *testing.T) {
	if !avx2MaskingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	runALiBiBiasParity(t, ALiBiBiasF32AVX2, ALiBiBiasFloat32AVX2Asm, 0x3820)
}

func TestALiBiBiasF32SSE2Parity(t *testing.T) {
	if !sse2MaskingAvailable() {
		t.Skip("SSE2 required")
	}

	runALiBiBiasParity(t, ALiBiBiasF32SSE2, ALiBiBiasFloat32SSE2Asm, 0x3821)
}

func runALiBiBiasParity(
	testingObject *testing.T,
	runWrapper func(*float32, *float32, *float32, int, int),
	runAsm func(*float32, *float32, *float32, int, int),
	seedBase int64,
) {
	convey.Convey("Given ALiBi bias SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match alibiBiasF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, seedBase+int64(length))
				slope := []float32{0.25}
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				alibiBiasF32Generic(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&want[0]),
					side,
					side,
				)
				runWrapper(&scores[0], &slope[0], &got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match alibiBiasF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, seedBase+0x100+int64(length))
				slope := []float32{0.5}
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				alibiBiasF32Generic(
					unsafe.Pointer(&scores[0]),
					unsafe.Pointer(&slope[0]),
					unsafe.Pointer(&want[0]),
					side,
					side,
				)
				runAsm(&scores[0], &slope[0], &got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			}
		})
	})
}

func runApplyMaskParity(
	testingObject *testing.T,
	runWrapper func(*float32, *float32, *float32, int),
	runAsm func(*float32, *float32, *float32, int),
	seedBase int64,
) {
	convey.Convey("Given apply-mask SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match applyMaskF32Generic for N=%d", length), func() {
				input := randomMaskingFloat32(length, seedBase+int64(length))
				mask := randomMaskingFloat32(length, seedBase+0x10+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				runWrapper(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match applyMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMaskingFloat32(length, seedBase+0x100+int64(length))
				mask := randomMaskingFloat32(length, seedBase+0x110+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				runAsm(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			}
		})
	})
}

func runCausalMaskParity(
	testingObject *testing.T,
	runWrapper func(*float32, int, int),
	runAsm func(*float32, int, int),
) {
	convey.Convey("Given causal-mask SIMD kernel", testingObject, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match causalMaskF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				runWrapper(&got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			})
		}

		convey.Convey("It should match causalMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				runAsm(&got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maskingAVX512MaxULP)
			}
		})
	})
}
