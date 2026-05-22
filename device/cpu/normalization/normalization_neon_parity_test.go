//go:build arm64

package normalization

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const normalizationNEONMaxULP = 1

func assertSquaredDiffSumParity(
	testingTB *testing.T,
	got, want float32,
	length int,
) {
	testingTB.Helper()

	tolerance := math.Max(math.Abs(float64(want)), 1.0) * float64(length) * 0x1p-50

	if math.Abs(float64(got-want)) > tolerance {
		testingTB.Fatalf(
			"N=%d got=%g want=%g diff=%g tol=%g",
			length, got, want, got-want, tolerance,
		)
	}
}

func TestNormSquaredDiffSumF32NEONParity(t *testing.T) {
	convey.Convey("Given NormSquaredDiffSum float32 NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row := randomNormalizationRow(length, 0x5F00+int64(length))
				mean := float32(0.125)

				want := NormSquaredDiffSumGeneric(row, mean)
				got := normSquaredDiffSumF32NEON(row, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomNormalizationRow(length, 0x5F01+int64(length))
				mean := float32(-0.25)

				want := NormSquaredDiffSumGeneric(row, mean)
				got := NormSquaredDiffSumFloat32NEONAsm(&row[0], length, mean)

				assertSquaredDiffSumParity(t, got, want, length)
			}
		})
	})
}

func TestNormApplyConstScaleBiasF32NEONParity(t *testing.T) {
	convey.Convey("Given NormApplyConstScaleBias float32 NEON", t, func() {
		mean := float32(0.05)
		invStdDev := float32(1.25)
		scale := float32(0.9)
		bias := float32(-0.1)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for N=%d", length), func() {
				row := randomNormalizationRow(length, 0x5F02+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				NormApplyConstScaleBiasGeneric(want, row, mean, invStdDev, scale, bias)
				normApplyConstScaleBiasF32NEON(got, row, mean, invStdDev, scale, bias)

				parity.AssertFloat32SlicesWithinULP(t, got, want, normalizationNEONMaxULP)
			})
		}

		convey.Convey("It should match generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				row := randomNormalizationRow(length, 0x5F03+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				NormApplyConstScaleBiasGeneric(want, row, mean, invStdDev, scale, bias)
				NormApplyConstScaleBiasFloat32NEONAsm(
					&got[0], &row[0], length,
					mean, invStdDev, scale, bias,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, normalizationNEONMaxULP)
			}
		})
	})
}
