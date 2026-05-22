//go:build arm64

package active_inference

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func assertScalarSumParityNEON(
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

func TestFreeEnergyF32NEONParity(t *testing.T) {
	convey.Convey("Given FreeEnergyF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FreeEnergyF32Generic for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA300+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], length)
				got := FreeEnergyF32NEON(&likelihood[0], &posterior[0], &prior[0], length)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestExpectedFreeEnergyF32NEONParity(t *testing.T) {
	convey.Convey("Given ExpectedFreeEnergyF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ExpectedFreeEnergyF32Generic for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA310+int64(length),
				)

				want := ExpectedFreeEnergyF32Generic(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)
				got := ExpectedFreeEnergyF32NEON(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestBeliefUpdateF32NEONParity(t *testing.T) {
	convey.Convey("Given BeliefUpdateF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match BeliefUpdateF32Generic for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA320+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateF32NEON(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match BeliefUpdateF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA321+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateFloat32NEONAsm(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestPrecisionWeightF32NEONParity(t *testing.T) {
	convey.Convey("Given PrecisionWeightF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PrecisionWeightF32Generic for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA330+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightF32NEON(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match PrecisionWeightF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA331+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightFloat32NEONAsm(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		})
	})
}

func TestFreeEnergyFloat32NEONAsmBlockParity(t *testing.T) {
	convey.Convey("Given FreeEnergyFloat32NEONAsm on 4-wide blocks", t, func() {
		for _, length := range parity.Lengths {
			blockCount := length &^ 3

			if blockCount == 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("It should match generic prefix for block N=%d", blockCount), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA340+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], blockCount)
				got := FreeEnergyFloat32NEONAsm(&likelihood[0], &posterior[0], &prior[0], blockCount)

				assertScalarSumParityNEON(t, got, want, blockCount)
			})
		}
	})
}
