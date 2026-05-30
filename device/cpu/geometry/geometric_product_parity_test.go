package geometry

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGeometricProductNativeParity(t *testing.T) {
	Convey("Given random multivector pairs", t, func() {
		randomSource := rand.New(rand.NewSource(17))

		for pairIndex := range 256 {
			left := randomMultivector(randomSource)
			right := randomMultivector(randomSource)

			var nativeResult Multivector
			var referenceResult Multivector

			geometricProductFloat64Scalar(&left[0], &right[0], &referenceResult[0])
			geometricProductKernel(&left[0], &right[0], &nativeResult[0])

			for componentIndex := range 8 {
				So(
					nativeResult[componentIndex],
					ShouldAlmostEqual,
					referenceResult[componentIndex],
					1e-14,
				)
			}

			_ = pairIndex
		}
	})
}

func TestRotorSimilarityNativeParity(t *testing.T) {
	Convey("Given random phase rotors", t, func() {
		randomSource := rand.New(rand.NewSource(23))

		leftRotor := make(PhaseRotor, PhaseDialDimensions)
		rightRotor := make(PhaseRotor, PhaseDialDimensions)

		for rotorIndex := range PhaseDialDimensions {
			leftRotor[rotorIndex] = randomMultivector(randomSource)
			rightRotor[rotorIndex] = randomMultivector(randomSource)
		}

		nativeSimilarity := rotorSimilarityAverage(leftRotor, rightRotor)
		referenceSimilarity := rotorSimilarity128Scalar(&leftRotor[0][0], &rightRotor[0][0], len(leftRotor))

		So(nativeSimilarity, ShouldAlmostEqual, referenceSimilarity, 1e-14)
	})
}

func BenchmarkGeometricProductNative(b *testing.B) {
	left := Rotor(0.7, 0.577, 0.577, 0.577)
	right := Rotor(1.2, 0, 0.707, 0.707)
	var result Multivector

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		geometricProductKernel(&left[0], &right[0], &result[0])
	}
}

func BenchmarkRotorSimilarity128Native(b *testing.B) {
	randomSource := rand.New(rand.NewSource(41))
	leftRotor := make(PhaseRotor, PhaseDialDimensions)
	rightRotor := make(PhaseRotor, PhaseDialDimensions)

	for rotorIndex := range PhaseDialDimensions {
		leftRotor[rotorIndex] = randomMultivector(randomSource)
		rightRotor[rotorIndex] = randomMultivector(randomSource)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_ = rotorSimilarityAverage(leftRotor, rightRotor)
	}
}

func randomMultivector(randomSource *rand.Rand) Multivector {
	return Multivector{
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
		randomSource.Float64()*2 - 1,
	}
}

func TestGeometricProductReferenceMatchesLegacy(t *testing.T) {
	Convey("Given basis multivectors", t, func() {
		identity := Multivector{1, 0, 0, 0, 0, 0, 0, 0}
		basisE12 := Multivector{0, 0, 0, 0, 1, 0, 0, 0}

		var reference [8]float64

		geometricProductFloat64Scalar(&identity[0], &basisE12[0], &reference[0])

		So(reference[4], ShouldAlmostEqual, 1.0, math.SmallestNonzeroFloat64)
	})
}
