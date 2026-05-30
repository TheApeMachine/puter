package geometry

import (
	"context"
	"testing"
	"unsafe"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device"
)

func TestGeometryDeviceOps(t *testing.T) {
	Convey("Given device.Geometry CPU implementation", t, func() {
		geometryBackend := New()

		Convey("GeometricProduct matches reference", func() {
			left := Multivector{1, 0, 0, 0, 1, 0, 0, 0}
			right := Rotor(0.5, 0, 0, 1)
			var nativeResult Multivector
			var referenceResult Multivector

			geometricProductFloat64Scalar(&left[0], &right[0], &referenceResult[0])
			geometryBackend.GeometricProduct(
				unsafe.Pointer(&nativeResult[0]),
				unsafe.Pointer(&left[0]),
				unsafe.Pointer(&right[0]),
			)

			for componentIndex := range 8 {
				So(
					nativeResult[componentIndex],
					ShouldAlmostEqual,
					referenceResult[componentIndex],
					1e-14,
				)
			}
		})

		Convey("PhaseDialSimilarity writes scalar destination", func() {
			left := NewPhaseDial().EncodeFromValues([]ValueToken{ValueTokenFromBytes([]byte("a"))})
			right := NewPhaseDial().EncodeFromValues([]ValueToken{ValueTokenFromBytes([]byte("b"))})
			var destination float64

			geometryBackend.PhaseDialSimilarity(
				unsafe.Pointer(&destination),
				unsafe.Pointer(&left[0]),
				unsafe.Pointer(&right[0]),
			)

			So(destination, ShouldAlmostEqual, left.Similarity(right), 1e-12)
		})

		Convey("EigenToroidalFromTags matches EigenModeToroidal", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			reference, err := NewEigenModeToroidal(
				EigenWithContext(ctx),
				func(toroidal *EigenModeToroidal) {
					toroidal.cancel = cancel
					toroidal.affinity = []uint64{1}
				},
			)
			So(err, ShouldBeNil)

			tags := []uint64{'a', 'b', 'a', 'b', 'a', 'b'}
			reference.BuildCooccurrenceFromWords(tags, 2)

			var (
				nativePhase     [512]float64
				nativeFrequency [512]float64
			)

			geometryBackend.EigenToroidalFromTags(
				unsafe.Pointer(&nativePhase[0]),
				unsafe.Pointer(&nativeFrequency[0]),
				unsafe.Pointer(&tags[0]),
				len(tags),
				2,
			)

			for index := range device.EigenSymbolDimensions {
				So(nativePhase[index], ShouldAlmostEqual, reference.phase[index], 1e-12)
				So(nativeFrequency[index], ShouldAlmostEqual, reference.frequency[index], 1e-12)
			}
		})

		Convey("EigenCircularMeanPhase writes scalar destination", func() {
			var phaseTable [512]float64

			for index := range phaseTable {
				phaseTable[index] = float64(index) / 512.0
			}

			sequence := []byte("ab")
			var destination float64

			geometryBackend.EigenCircularMeanPhase(
				unsafe.Pointer(&destination),
				unsafe.Pointer(&phaseTable[0]),
				unsafe.Pointer(&sequence[0]),
				len(sequence),
			)

			expected, err := SeqCircularMeanPhaseFromPhases(&phaseTable, sequence)
			So(err, ShouldBeNil)
			So(destination, ShouldAlmostEqual, expected, 1e-12)
		})
	})
}
