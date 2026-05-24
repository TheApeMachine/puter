//go:build xla

package dropout_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpudropout "github.com/theapemachine/puter/device/cpu/dropout"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceDropout = cpudropout.New()

func TestDropoutXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA dropout", t, func() {
		config := device.DropoutConfig{Rate: 0.25, Seed: 0xC0FFEE}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := xlaparity.RandomUnaryInput(count, 0xD400+int64(count))
				want := make([]float32, count)
				referenceDropout.Dropout(
					unsafe.Pointer(&want[0]),
					unsafe.Pointer(&source[0]),
					count,
					config,
					dtype.Float32,
				)

				sourceTensor := harness.UploadVector(source, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer sourceTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Dropout(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(sourceTensor),
					count,
					config,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
