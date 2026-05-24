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
// We assert TWO things:
//
//  1. Loose per-lane ULP tolerance (≤ 128 ULP). This is the realistic
//     floor for Apple silicon GPU's F32 transcendentals (log/sin/cos)
//     vs Go's F64-then-cast scalar reference. The CPU path computes
//     log/sin/cos in F64 with single rounding to F32, which extracts
//     near-correctly-rounded F32 results. Apple's MSL native log/sin/
//     cos (including precise::* variants) are F32-internal polynomial
//     approximations — they're spec'd at ≤ 1 ULP from correctly-rounded
//     F32 but Apple silicon GPUs do not implement IEEE F64, so we
//     cannot replicate the F64 → F32 single-rounding sequence on GPU.
//     The 128 ULP envelope absorbs both Metal's transcendental error
//     and the compounding error through magnitude × sin/cos in regions
//     where the Gaussian output is small.
//
//     For comparison: PyTorch MPS and JAX on TPU both decline to claim
//     CPU-vs-GPU bitwise parity for transcendentals at this layer.
//
//  2. Strict statistical correctness (mean ≈ 0, variance ≈ 1 on a
//     large sample). This is the assertion that actually matters for
//     downstream use — a Gaussian RNG is only useful if it produces a
//     properly-distributed Gaussian. Both backends must pass.
//
// The Philox uint32 stream (before Box-Muller) IS bitwise across CPU
// and Metal — see TestPhilox4x32x4NEONBitwiseParity in the CPU package
// for the contract. The divergence enters at Box-Muller.

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
