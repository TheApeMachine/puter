//go:build darwin && cgo

package random

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpurandom "github.com/theapemachine/puter/device/cpu/random"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

// Metal random_normal_float32 parity vs CPU scalar reference.
//
// Bitwise parity is NOT expected on the final Gaussian outputs because
// Metal's native single-precision log/sin/cos do not bit-match Go's
// F64-then-cast scalar reference. The kernel is bitwise on the Philox
// uint32 stream and on the uniform conversion; the divergence enters
// at the transcendental approximations.
//
// We assert ≤ 8 ULP per lane, in line with the existing Metal tolerance
// for activation kernels that involve transcendentals (GeluTanh: 8 ULP).
// If the actual divergence on real hardware lands tighter, we can
// narrow the tolerance.

const randomNormalULP = 8

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

			convey.Convey("It matches the CPU scalar within tolerance for "+testCase.name, func() {
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
