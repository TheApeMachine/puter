//go:build arm64

package masking

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const maskingNEONMaxULP = 0

func TestApplyMaskF32NEONParity(t *testing.T) {
	convey.Convey("Given ApplyMaskF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match applyMaskF32Generic for N=%d", length), func() {
				input := randomMaskingFloat32(length, 0x1920+int64(length))
				mask := randomMaskingFloat32(length, 0x1921+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				ApplyMaskF32NEON(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			})
		}

		convey.Convey("It should match applyMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				input := randomMaskingFloat32(length, 0x1922+int64(length))
				mask := randomMaskingFloat32(length, 0x1923+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				applyMaskF32Generic(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&mask[0]),
					unsafe.Pointer(&want[0]),
					length,
				)
				ApplyMaskFloat32NEONAsm(&input[0], &mask[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			}
		})
	})
}

func TestCausalMaskF32NEONParity(t *testing.T) {
	convey.Convey("Given CausalMaskF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match causalMaskF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				CausalMaskF32NEON(&got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			})
		}

		convey.Convey("It should match causalMaskF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				want := make([]float32, side*side)
				got := make([]float32, side*side)

				causalMaskF32Generic(unsafe.Pointer(&want[0]), side, side)
				for rowIndex := 0; rowIndex < side; rowIndex++ {
					zeroCount := rowIndex + 1
					if zeroCount > side {
						zeroCount = side
					}

					infCount := side - zeroCount
					rowOutput := &got[rowIndex*side]
					causalMaskFloat32NEONFillAsm(rowOutput, zeroCount, infCount)
				}

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			}
		})
	})
}

func TestALiBiBiasF32NEONParity(t *testing.T) {
	convey.Convey("Given ALiBiBiasF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match alibiBiasF32Generic for N=%d", length), func() {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, 0x1930+int64(length))
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
				ALiBiBiasF32NEON(&scores[0], &slope[0], &got[0], side, side)

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			})
		}

		convey.Convey("It should match alibiBiasF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				side := maskingSquareSide(length)
				scores := randomMaskingScores(side, side, 0x1931+int64(length))
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
				for rowIndex := 0; rowIndex < side; rowIndex++ {
					for colIndex := 0; colIndex < side; colIndex++ {
						index := rowIndex*side + colIndex
						alibiBiasFloat32NEONElemAsm(
							&scores[index],
							&slope[0],
							&got[index],
							rowIndex-colIndex,
						)
					}
				}

				parity.AssertFloat32SlicesWithinULP(t, got, want, maskingNEONMaxULP)
			}
		})
	})
}
