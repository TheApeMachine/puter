package geometry

import (
	"math"
	"math/cmplx"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestPhaseDialPrimeTable(t *testing.T) {
	Convey("Given PhaseDialPrimes", t, func() {
		So(len(PhaseDialPrimes), ShouldEqual, PhaseDialDimensions)
		So(PhaseDialPrimes[0], ShouldEqual, 2)
		So(PhaseDialPrimes[1], ShouldEqual, 3)
		So(PhaseDialPrimes[2], ShouldEqual, 5)
	})
}

func TestValueTokenFromBytes(t *testing.T) {
	Convey("Given ValueTokenFromBytes", t, func() {
		token := ValueTokenFromBytes([]byte("abc"))

		So(token.Word(0)&0xff, ShouldEqual, 'a')
		So((token.Word(0)>>8)&0xff, ShouldEqual, 'b')
		So((token.Word(0)>>16)&0xff, ShouldEqual, 'c')
	})
}

func TestNewPhaseDial(t *testing.T) {
	Convey("Given NewPhaseDial", t, func() {
		Convey("It should return a zeroed dial of PhaseDialDimensions length", func() {
			dial := NewPhaseDial()
			So(dial, ShouldNotBeNil)
			So(len(dial), ShouldEqual, PhaseDialDimensions)

			for _, val := range dial {
				So(real(val), ShouldEqual, 0)
				So(imag(val), ShouldEqual, 0)
			}
		})
	})
}

func TestPhaseDialEncodeFromValues(t *testing.T) {
	Convey("Given PhaseDial encoding", t, func() {
		dial := NewPhaseDial()

		Convey("When encoding an empty sequence", func() {
			encoded := dial.EncodeFromValues(nil)
			So(encoded, ShouldNotBeNil)
		})

		Convey("When encoding a single value", func() {
			token := ValueTokenFromBytes([]byte("a"))
			encoded := NewPhaseDial().EncodeFromValues([]ValueToken{token})
			var magnitude float64

			for _, val := range encoded {
				realPart, imagPart := real(val), imag(val)
				magnitude += realPart*realPart + imagPart*imagPart
			}

			So(math.Sqrt(magnitude), ShouldAlmostEqual, 1.0, 0.0001)
			So(encoded[0], ShouldNotEqual, complex(0, 0))
		})
	})
}

func TestPhaseDialSimilarity(t *testing.T) {
	Convey("Given distinct payloads", t, func() {
		seqA := make([]byte, 50)

		for index := range seqA {
			seqA[index] = 'a'
		}

		seqB := make([]byte, 50)

		for index := range seqB {
			seqB[index] = 'b'
		}

		encodedA := NewPhaseDial().EncodeFromValues([]ValueToken{ValueTokenFromBytes(seqA)})
		encodedB := NewPhaseDial().EncodeFromValues([]ValueToken{ValueTokenFromBytes(seqB)})

		differences := 0

		for index := range encodedA {
			if cmplx.Abs(encodedA[index]-encodedB[index]) > 0.001 {
				differences++
			}
		}

		So(differences, ShouldBeGreaterThan, 100)

		similarity := encodedA.Similarity(encodedB)
		So(similarity, ShouldBeBetweenOrEqual, -1, 1)
		So(similarity, ShouldNotAlmostEqual, 1.0, 0.01)
	})
}

func TestNewPhaseRotor(t *testing.T) {
	Convey("Given NewPhaseRotor", t, func() {
		rotor := NewPhaseRotor()
		So(len(rotor), ShouldEqual, PhaseDialDimensions)

		for _, multivector := range rotor {
			for _, component := range multivector {
				So(component, ShouldEqual, 0)
			}
		}

		encoded := NewPhaseRotor().EncodeFromValues([]ValueToken{ValueTokenFromBytes([]byte("rotor"))})
		So(len(encoded), ShouldEqual, PhaseDialDimensions)

		selfSimilarity := encoded.Similarity(encoded)
		So(selfSimilarity, ShouldAlmostEqual, 1.0, 0.0001)

		dial := encoded.ToDialCompat()
		So(len(dial), ShouldEqual, PhaseDialDimensions)

		var magnitude float64

		for _, val := range dial {
			realPart, imagPart := real(val), imag(val)
			magnitude += realPart*realPart + imagPart*imagPart
		}

		So(math.Sqrt(magnitude), ShouldAlmostEqual, 1.0, 0.0001)
	})
}

func BenchmarkNewPhaseDial(benchmark *testing.B) {
	benchmark.ReportAllocs()

	for benchmark.Loop() {
		_ = NewPhaseDial()
	}
}

func BenchmarkPhaseDialEncodeFromValues(benchmark *testing.B) {
	token := ValueTokenFromBytes([]byte("benchmark value sequence for phase encoding"))
	payload := []ValueToken{token}
	dial := NewPhaseDial()

	benchmark.ResetTimer()
	benchmark.ReportAllocs()

	for benchmark.Loop() {
		_ = dial.EncodeFromValues(payload)
	}
}
