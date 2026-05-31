//go:build darwin && cgo

package random

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpurandom "github.com/theapemachine/puter/device/cpu/random"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

// Metal random_normal_float32 parity vs CPU scalar reference.
//
// Philox output is bitwise equivalent to NormalFloat32Scalar. Box-Muller
// uses Go's float64 log/sqrt/sincos sequence on CPU; Metal uses precise
// float32 transcendentals because MSL has no native double type. Empirical
// per-lane divergence is ≤ 128 ULP on Apple silicon while the distribution
// test still validates mean/variance.

const randomNormalULP = 128

func TestRandomNormalFloat32MetalParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given the Metal random_normal_float32 kernel", t, func() {
		cases := []struct {
			name    string
			seed    uint64
			counter uint64
		}{
			{"AllZero", 0, 0},
			{"SmallSeed", 0xDEADBEEFCAFEBABE, 0x1000},
			{"LargeCtr", 0xA4093822299F31D0, 0x0000000000010000},
			{"KAT-mixed", 0x299F31D0A4093822, 0x85A308D3243F6A88},
		}

		for _, testCase := range cases {
			testCase := testCase

			convey.Convey("It matches the CPU scalar within 128 ULP for "+testCase.name, func() {
				for _, count := range []int{4, 8, 64, 1024, 8192} {
					count := count

					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						want := make([]float32, count)
						cpurandom.NormalFloat32Scalar(want, count, testCase.seed, testCase.counter)

						destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
						defer destinationTensor.Close()

						if err := DispatchNormalRefs(
							harness.ContextRef(),
							destinationTensor.Ref(),
							uint32(count),
							testCase.seed,
							testCase.counter,
						); err != nil {
							t.Fatalf("dispatch random_normal_float32: %v", err)
						}

						got := harness.DownloadFloat32(destinationTensor, dtype.Float32)
						parity.AssertFloat32SlicesWithinULP(t, got, want, randomNormalULP)
					})
				}
			})
		}
	})
}

/*
TestRandomNormalFloat32MetalDistribution asserts the property that
actually matters for a Gaussian RNG: the produced sample is a properly-
distributed standard normal. This holds independently of any backend's
transcendental precision, because the assertion is on the bulk
statistical behavior over 65536 samples, not on individual lanes.

Both the CPU scalar reference and the Metal kernel must pass this
test; if either fails, the RNG is broken regardless of how close
per-lane parity is.
*/
func TestRandomNormalFloat32MetalDistribution(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given a Metal-produced sample of 65536 standard normals", t, func() {
		const count = 1 << 16
		const seed = uint64(0xC0FFEE)
		const counter = uint64(0)

		destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
		defer destinationTensor.Close()

		if err := DispatchNormalRefs(
			harness.ContextRef(),
			destinationTensor.Ref(),
			uint32(count),
			seed,
			counter,
		); err != nil {
			t.Fatalf("dispatch random_normal_float32: %v", err)
		}

		samples := harness.DownloadFloat32(destinationTensor, dtype.Float32)

		var sum, sumSquared float64
		for _, value := range samples {
			sum += float64(value)
			sumSquared += float64(value) * float64(value)
		}

		mean := sum / float64(count)
		variance := sumSquared/float64(count) - mean*mean

		convey.Convey("It has empirical mean near 0 and variance near 1", func() {
			// 65536 standard-normal samples: 99.7% bound on mean is
			// ±3/sqrt(N) ≈ ±0.012. Use 0.05 as a loose deterministic
			// bound that catches gross distributional bugs without
			// flaking on the specific seed.
			convey.So(math.Abs(mean), convey.ShouldBeLessThan, 0.05)
			convey.So(math.Abs(variance-1.0), convey.ShouldBeLessThan, 0.05)
		})
	})
}
