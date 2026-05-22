//go:build amd64

package active_inference

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx2ActiveInferenceAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2ActiveInferenceAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestFreeEnergyF32AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given FreeEnergyF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FreeEnergyF32Generic for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA250+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], length)
				got := FreeEnergyF32AVX2(&likelihood[0], &posterior[0], &prior[0], length)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestFreeEnergyF32SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given FreeEnergyF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match FreeEnergyF32Generic for N=%d", length), func() {
				likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA260+int64(length))

				want := FreeEnergyF32Generic(&likelihood[0], &posterior[0], &prior[0], length)
				got := FreeEnergyF32SSE2(&likelihood[0], &posterior[0], &prior[0], length)

				parity.AssertFloat32SlicesWithinULP(
					t, []float32{got}, []float32{want}, activeInferenceLogMaxULP,
				)
			})
		}
	})
}

func TestExpectedFreeEnergyF32AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given ExpectedFreeEnergyF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ExpectedFreeEnergyF32Generic for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA270+int64(length),
				)

				want := ExpectedFreeEnergyF32Generic(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)
				got := ExpectedFreeEnergyF32AVX2(
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

func TestExpectedFreeEnergyF32SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given ExpectedFreeEnergyF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ExpectedFreeEnergyF32Generic for N=%d", length), func() {
				predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(
					length, 0xA280+int64(length),
				)

				want := ExpectedFreeEnergyF32Generic(
					&predictedObs[0], &preferredObs[0], &predictedState[0],
					length, length,
				)
				got := ExpectedFreeEnergyF32SSE2(
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

func TestBeliefUpdateF32AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given BeliefUpdateF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match BeliefUpdateF32Generic for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA290+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateF32AVX2(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match BeliefUpdateF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA291+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateFloat32AVX2Asm(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestBeliefUpdateF32SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given BeliefUpdateF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match BeliefUpdateF32Generic for N=%d", length), func() {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA2A0+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateF32SSE2(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match BeliefUpdateF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				likelihood, prior, _ := randomActiveInferenceVectors(length, 0xA2A1+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], length)
				BeliefUpdateFloat32SSE2Asm(&likelihood[0], &prior[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestPrecisionWeightF32AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given PrecisionWeightF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PrecisionWeightF32Generic for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA2B0+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightF32AVX2(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match PrecisionWeightF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA2B1+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightFloat32AVX2Asm(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}

func TestPrecisionWeightF32SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given PrecisionWeightF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PrecisionWeightF32Generic for N=%d", length), func() {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA2C0+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightF32SSE2(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			})
		}

		convey.Convey("It should match PrecisionWeightF32Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				errors, precision, _ := randomActiveInferenceVectors(length, 0xA2C1+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				PrecisionWeightF32Generic(&errors[0], &precision[0], &want[0], length)
				PrecisionWeightFloat32SSE2Asm(&errors[0], &precision[0], &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, activeInferenceScalarMaxULP)
			}
		})
	})
}
