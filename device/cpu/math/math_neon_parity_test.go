//go:build arm64

package math

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const mathNEONMaxULP = 2

func TestInvSqrtDimScaleF32NEONParity(t *testing.T) {
	convey.Convey("Given InvSqrtDimScaleF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match InvSqrtDimScaleGeneric for N=%d", length), func() {
				input := randomMathFloat32(length, 0x3210+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				InvSqrtDimScaleF32NEON(got, input, 64)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}

		convey.Convey("It should match InvSqrtDimScaleGeneric via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMathFloat32(length, 0x3211+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				InvSqrtDimScaleGeneric(want, input, 64)
				scale := float32(1.0 / 8.0)
				InvSqrtDimScaleFloat32NEONAsm(&got[0], &input[0], scale, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			}
		})
	})
}

func TestLogSumExpRowF32NEONParity(t *testing.T) {
	convey.Convey("Given logSumExpRowF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpRowGeneric for N=%d", length), func() {
				row := randomMathFloat32(length, 0x3212+int64(length))
				want := LogSumExpRowGeneric(row)
				got := logSumExpRowF32NEON(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathNEONMaxULP)
			})
		}

		convey.Convey("It should match LogSumExpRowGeneric via direct asm parts at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomMathFloat32(length, 0x3213+int64(length))
				want := LogSumExpRowGeneric(row)
				got := logSumExpRowF32NEON(row)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, mathNEONMaxULP)
			})
		})
	})
}

func TestLogSumExpF32NEONParity(t *testing.T) {
	convey.Convey("Given LogSumExpF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match LogSumExpGeneric for N=%d", length), func() {
				cols := mathSquareSide(length)
				rows := length / cols

				if rows < 1 {
					rows = 1
				}

				input := randomMathFloat32(rows*cols, 0x3214+int64(length))
				want := make([]float32, rows)
				got := make([]float32, rows)

				LogSumExpGeneric(input, cols, want)
				LogSumExpF32NEON(input, cols, got)

				parity.AssertFloat32SlicesWithinULP(t, got, want, mathNEONMaxULP)
			})
		}
	})
}

func TestOuterF32NEONParity(t *testing.T) {
	convey.Convey("Given OuterF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match OuterGeneric for N=%d", length), func() {
				leftLen := mathSquareSide(length)
				rightLen := length / leftLen

				if rightLen < 1 {
					rightLen = 1
				}

				left := randomMathFloat32(leftLen, 0x3215+int64(length))
				right := randomMathFloat32(rightLen, 0x3216+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterF32NEON(left, right, got)

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

				left := randomMathFloat32(leftLen, 0x3217+int64(length))
				right := randomMathFloat32(rightLen, 0x3218+int64(length))
				want := make([]float32, leftLen*rightLen)
				got := make([]float32, leftLen*rightLen)

				OuterGeneric(left, right, want)
				OuterFloat32NEONAsm(&got[0], &left[0], &right[0], leftLen, rightLen)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		})
	})
}
