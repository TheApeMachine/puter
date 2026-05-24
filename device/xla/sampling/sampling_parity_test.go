//go:build xla

package sampling_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
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
				var want int32
				referenceSampling.GreedySample(
					unsafe.Pointer(&want),
					unsafe.Pointer(&logits[0]),
					count,
					dtype.Float32,
				)

				logitsTensor := harness.UploadVector(logits, dtype.Float32)
				defer logitsTensor.Close()

				var got int32
				harness.Backend().GreedySample(
					unsafe.Pointer(&got),
					xla.ResidentPointer(logitsTensor),
					count,
					dtype.Float32,
				)

				convey.So(got, convey.ShouldEqual, want)
			})
		}
	})
}

func TestTopKSampleXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA TopK sample", t, func() {
		config := device.SamplingConfig{
			Temperature: 0.8,
			TopK:        8,
			Seed:        0x5151,
		}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				logits := xlaparity.RandomUnaryInput(count, 0x5600+int64(count))
				var want int32
				referenceSampling.TopKSample(
					unsafe.Pointer(&want),
					unsafe.Pointer(&logits[0]),
					count,
					config,
					dtype.Float32,
				)

				logitsTensor := harness.UploadVector(logits, dtype.Float32)
				defer logitsTensor.Close()

				var got int32
				harness.Backend().TopKSample(
					unsafe.Pointer(&got),
					xla.ResidentPointer(logitsTensor),
					count,
					config,
					dtype.Float32,
				)

				convey.So(got, convey.ShouldEqual, want)
			})
		}
	})
}

func TestTopPSampleXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA TopP sample", t, func() {
		config := device.SamplingConfig{
			Temperature: 1.0,
			TopP:        0.85,
			Seed:        0x5252,
		}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				logits := xlaparity.RandomUnaryInput(count, 0x5700+int64(count))
				var want int32
				referenceSampling.TopPSample(
					unsafe.Pointer(&want),
					unsafe.Pointer(&logits[0]),
					count,
					config,
					dtype.Float32,
				)

				logitsTensor := harness.UploadVector(logits, dtype.Float32)
				defer logitsTensor.Close()

				var got int32
				harness.Backend().TopPSample(
					unsafe.Pointer(&got),
					xla.ResidentPointer(logitsTensor),
					count,
					config,
					dtype.Float32,
				)

				convey.So(got, convey.ShouldEqual, want)
			})
		}
	})
}
