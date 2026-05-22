//go:build amd64

package math

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const mathReducedMaxULP = 2

func avx2MathAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2MathAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestInvSqrtDimScaleF32AVX2Parity(t *testing.T) {
	if !avx2MathAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given InvSqrtDimScaleF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match InvSqrtDimScaleGeneric for N=%d", length), func() {
				input := randomMathFloat32(length, 0x2310+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				InvSqrtDimScaleF32AVX2(got, input, 64)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match InvSqrtDimScaleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMathFloat32(length, 0x2311+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				scale := float32(1.0 / 8.0)
				InvSqrtDimScaleFloat32AVX2Asm(&got[0], &input[0], scale, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestInvSqrtDimScaleF32SSE2Parity(t *testing.T) {
	if !sse2MathAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given InvSqrtDimScaleF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match InvSqrtDimScaleGeneric for N=%d", length), func() {
				input := randomMathFloat32(length, 0x2320+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				InvSqrtDimScaleF32SSE2(got, input, 64)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match InvSqrtDimScaleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMathFloat32(length, 0x2321+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				scale := float32(1.0 / 8.0)
				InvSqrtDimScaleFloat32SSE2Asm(&got[0], &input[0], scale, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestLogSumExpRowF32AVX2Parity(t *testing.T) {
	if !avx2MathAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given logSumExpRowF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpRowGeneric for N=%d", length), func() {
				row := randomMathFloat32(length, 0x2330+int64(length))
				want := LogSumExpRowGeneric(row)
				got := logSumExpRowF32AVX2(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathReducedMaxULP)
			})
		}

		convey.Convey("It should match LogSumExpRowGeneric via direct asm parts at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomMathFloat32(length, 0x2331+int64(length))
				want := LogSumExpRowGeneric(row)

				var maximum float32
				var expSum float32

				LogSumExpRowPartsFloat32AVX2Asm(&row[0], length, &maximum, &expSum)
				got := logSumExpRowF32AVX2(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathReducedMaxULP)
			}
		})
	})
}

func TestLogSumExpRowF32SSE2Parity(t *testing.T) {
	if !sse2MathAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given logSumExpRowF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpRowGeneric for N=%d", length), func() {
				row := randomMathFloat32(length, 0x2340+int64(length))
				want := LogSumExpRowGeneric(row)
				got := logSumExpRowF32SSE2(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathReducedMaxULP)
			})
		}

		convey.Convey("It should match LogSumExpRowGeneric via direct asm parts at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomMathFloat32(length, 0x2341+int64(length))
				want := LogSumExpRowGeneric(row)

				var maximum float32
				var expSum float32

				LogSumExpRowPartsFloat32SSE2Asm(&row[0], length, &maximum, &expSum)
				got := logSumExpRowF32SSE2(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathReducedMaxULP)
			}
		})
	})
}

func TestLogSumExpF32AVX2Parity(t *testing.T) {
	if !avx2MathAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given LogSumExpF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpGeneric for N=%d", length), func() {
				cols := mathSquareSide(length)
				rows := length / cols

				if rows < 1 {
					rows = 1
				}

				input := randomMathFloat32(rows*cols, 0x2350+int64(length))
				want := make([]float32, rows)
				got := make([]float32, rows)

				LogSumExpGeneric(input, cols, want)
				LogSumExpF32AVX2(input, cols, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, mathReducedMaxULP)
			})
		}
	})
}

func TestLogSumExpF32SSE2Parity(t *testing.T) {
	if !sse2MathAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given LogSumExpF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpGeneric for N=%d", length), func() {
				cols := mathSquareSide(length)
				rows := length / cols

				if rows < 1 {
					rows = 1
				}

				input := randomMathFloat32(rows*cols, 0x2360+int64(length))
				want := make([]float32, rows)
				got := make([]float32, rows)

				LogSumExpGeneric(input, cols, want)
				LogSumExpF32SSE2(input, cols, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, mathReducedMaxULP)
			})
		}
	})
}

func TestOuterF32AVX2Parity(t *testing.T) {
	if !avx2MathAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given OuterF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match OuterGeneric for N=%d", length), func() {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2370+int64(length))
				right := randomMathFloat32(rightLen, 0x2371+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterF32AVX2(left, right, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match OuterGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2372+int64(length))
				right := randomMathFloat32(rightLen, 0x2373+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterFloat32AVX2Asm(&got[0], &left[0], &right[0], leftLen, rightLen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestOuterF32SSE2Parity(t *testing.T) {
	if !sse2MathAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given OuterF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match OuterGeneric for N=%d", length), func() {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2380+int64(length))
				right := randomMathFloat32(rightLen, 0x2381+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterF32SSE2(left, right, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match OuterGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x2382+int64(length))
				right := randomMathFloat32(rightLen, 0x2383+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterFloat32SSE2Asm(&got[0], &left[0], &right[0], leftLen, rightLen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}
