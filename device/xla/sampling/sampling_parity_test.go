//go:build xla

package sampling_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpusampling "github.com/theapemachine/puter/device/cpu/sampling"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceSampling = cpusampling.New()

func TestGreedySampleXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA greedy sample", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				logits := xlaparity.RandomUnaryInput(count, 0x5500+int64(count))
				want := referenceSampling.GreedySample(
					unsafe.Pointer(&logits[0]),
					count,
					dtype.Float32,
				)

				logitsTensor := harness.UploadVector(logits, dtype.Float32)
				defer logitsTensor.Close()

				got := harness.Backend().GreedySample(
					xla.ResidentPointer(logitsTensor),
					count,
					dtype.Float32,
				)

				convey.So(got, convey.ShouldEqual, want)
			})
		}
	})
}
