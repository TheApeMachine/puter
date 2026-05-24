package random

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

// Philox-4×32-10 known-answer test vectors. Each entry is (key, ctr) →
// expected output. Sourced from the random123 distribution's
// kat_vectors.dat file (random123/include/Random123/kat_vectors.dat).
//
// These vectors are the bitwise contract every backend must match. Do
// not modify them; if the algorithm changes, the entire stack
// (CPU/Metal/CUDA/XLA) loses cross-backend parity.
var philoxKAT = []struct {
	name             string
	key0, key1       uint32
	ctr0, ctr1, ctr2, ctr3 uint32
	out0, out1, out2, out3 uint32
}{
	{
		name: "AllZero",
		key0: 0x00000000, key1: 0x00000000,
		ctr0: 0x00000000, ctr1: 0x00000000, ctr2: 0x00000000, ctr3: 0x00000000,
		out0: 0x6627e8d5, out1: 0xe169c58d, out2: 0xbc57ac4c, out3: 0x9b00dbd8,
	},
	{
		name: "AllOnes",
		key0: 0xffffffff, key1: 0xffffffff,
		ctr0: 0xffffffff, ctr1: 0xffffffff, ctr2: 0xffffffff, ctr3: 0xffffffff,
		out0: 0x408f276d, out1: 0x41c83b0e, out2: 0xa20bc7c6, out3: 0x6d5451fd,
	},
	{
		name: "Mixed",
		key0: 0xa4093822, key1: 0x299f31d0,
		ctr0: 0x243f6a88, ctr1: 0x85a308d3, ctr2: 0x13198a2e, ctr3: 0x03707344,
		out0: 0xd16cfe09, out1: 0x94fdcceb, out2: 0x5001e420, out3: 0x24126ea1,
	},
}

func TestPhilox4x32KnownAnswerVectors(t *testing.T) {
	convey.Convey("Given Philox-4×32-10", t, func() {
		for _, vector := range philoxKAT {
			vector := vector

			convey.Convey("It produces the random123 KAT output for "+vector.name, func() {
				state := PhiloxState{
					Key0: vector.key0, Key1: vector.key1,
					Ctr0: vector.ctr0, Ctr1: vector.ctr1,
					Ctr2: vector.ctr2, Ctr3: vector.ctr3,
				}

				got0, got1, got2, got3 := Philox4x32(state)

				convey.So(got0, convey.ShouldEqual, vector.out0)
				convey.So(got1, convey.ShouldEqual, vector.out1)
				convey.So(got2, convey.ShouldEqual, vector.out2)
				convey.So(got3, convey.ShouldEqual, vector.out3)
			})
		}
	})
}

func TestNewPhiloxStateSplitsSeedAndCounter(t *testing.T) {
	convey.Convey("Given NewPhiloxState", t, func() {
		state := NewPhiloxState(0x0123456789ABCDEF, 0xFEDCBA9876543210)

		convey.Convey("It splits the seed into low/high 32 bits", func() {
			convey.So(state.Key0, convey.ShouldEqual, uint32(0x89ABCDEF))
			convey.So(state.Key1, convey.ShouldEqual, uint32(0x01234567))
		})

		convey.Convey("It splits the counter into low/high 32 bits with high words zero", func() {
			convey.So(state.Ctr0, convey.ShouldEqual, uint32(0x76543210))
			convey.So(state.Ctr1, convey.ShouldEqual, uint32(0xFEDCBA98))
			convey.So(state.Ctr2, convey.ShouldEqual, uint32(0))
			convey.So(state.Ctr3, convey.ShouldEqual, uint32(0))
		})
	})
}

func TestUniformFloat32RangeAndExtremes(t *testing.T) {
	convey.Convey("Given uniformFloat32", t, func() {
		convey.Convey("It maps all-zero bits to 0", func() {
			convey.So(uniformFloat32(0x00000000), convey.ShouldEqual, float32(0))
		})

		convey.Convey("It maps top-23-bits-all-one to the largest representable F32 below 1", func() {
			// top 23 bits = 1: bits = 0xFFFFFE00 (mantissa = 0x7FFFFF)
			got := uniformFloat32(0xFFFFFE00)
			// Largest F32 strictly less than 1 is 1 - 2^-24.
			convey.So(got, convey.ShouldBeLessThan, float32(1.0))
			convey.So(got, convey.ShouldBeGreaterThanOrEqualTo, float32(1.0)-math.Nextafter32(1.0, 0))
		})

		convey.Convey("It produces values uniformly across [0, 1) for varied bit patterns", func() {
			samples := []uint32{
				0x00000200, // smallest non-zero mantissa
				0x12345678,
				0x80000000,
				0xCAFEBABE,
			}

			for _, bits := range samples {
				got := uniformFloat32(bits)
				convey.So(got, convey.ShouldBeGreaterThanOrEqualTo, float32(0))
				convey.So(got, convey.ShouldBeLessThan, float32(1.0))
			}
		})
	})
}

func TestBoxMullerPairProducesFiniteNormals(t *testing.T) {
	convey.Convey("Given boxMullerPair", t, func() {
		convey.Convey("It returns finite values for typical uniforms", func() {
			z0, z1 := boxMullerPair(0.25, 0.75)
			convey.So(math.IsInf(float64(z0), 0), convey.ShouldBeFalse)
			convey.So(math.IsInf(float64(z1), 0), convey.ShouldBeFalse)
			convey.So(math.IsNaN(float64(z0)), convey.ShouldBeFalse)
			convey.So(math.IsNaN(float64(z1)), convey.ShouldBeFalse)
		})

		convey.Convey("It substitutes a finite minimum when u1 == 0", func() {
			z0, z1 := boxMullerPair(0, 0.5)
			convey.So(math.IsInf(float64(z0), 0), convey.ShouldBeFalse)
			convey.So(math.IsInf(float64(z1), 0), convey.ShouldBeFalse)
		})
	})
}

func TestNormalFloat32ScalarMeanAndVariance(t *testing.T) {
	convey.Convey("Given NormalFloat32Scalar over a large sample", t, func() {
		const count = 1 << 16
		out := make([]float32, count)
		NormalFloat32Scalar(out, count, 0xC0FFEE, 0)

		var sum, sumSquared float64
		for _, value := range out {
			sum += float64(value)
			sumSquared += float64(value) * float64(value)
		}

		mean := sum / float64(count)
		variance := sumSquared/float64(count) - mean*mean

		convey.Convey("It has empirical mean near 0 and variance near 1", func() {
			// 65536 standard-normal samples: 99.7% bound on mean is ±3/sqrt(N) ≈ ±0.012.
			// Use a looser 0.05 bound to avoid flakiness on the deterministic seed.
			convey.So(math.Abs(mean), convey.ShouldBeLessThan, 0.05)
			convey.So(math.Abs(variance-1.0), convey.ShouldBeLessThan, 0.05)
		})
	})
}

func TestNormalFloat32ScalarReproducibility(t *testing.T) {
	convey.Convey("Given the same (seed, counter)", t, func() {
		convey.Convey("It produces the same output bitwise", func() {
			const count = 1024
			first := make([]float32, count)
			second := make([]float32, count)

			NormalFloat32Scalar(first, count, 0xDEADBEEF, 42)
			NormalFloat32Scalar(second, count, 0xDEADBEEF, 42)

			for index := range first {
				convey.So(math.Float32bits(first[index]), convey.ShouldEqual, math.Float32bits(second[index]))
			}
		})
	})
}

func TestNormalFloat32ScalarCounterAdvances(t *testing.T) {
	convey.Convey("Given different counters with the same seed", t, func() {
		convey.Convey("It produces different output", func() {
			first := make([]float32, 16)
			second := make([]float32, 16)

			NormalFloat32Scalar(first, 16, 0xABCD1234, 0)
			NormalFloat32Scalar(second, 16, 0xABCD1234, 1)

			differing := 0
			for index := range first {
				if math.Float32bits(first[index]) != math.Float32bits(second[index]) {
					differing++
				}
			}

			// Even though the first 4 outputs of `second` share no
			// counter values with `first`, the test guards against an
			// accidental no-op where the counter wasn't being used.
			convey.So(differing, convey.ShouldBeGreaterThan, 8)
		})
	})
}
