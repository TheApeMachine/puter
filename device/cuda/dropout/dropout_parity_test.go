//go:build cuda

package dropout

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpudropout "github.com/theapemachine/puter/device/cpu/dropout"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestDropoutCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA dropout", t, func() {
		config := device.DropoutConfig{Rate: 0.25, Seed: 0xC0FFEE}

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := parity.RandomUnaryInput(count, 0xD400+int64(count))
				want := make([]float32, count)
				seedWant := cpudropout.Default.DropoutSeedState(config.Seed)
				cpudropout.DropoutF32Generic(&want[0], &source[0], count, &seedWant, float32(1.0-config.Rate))

				sourceTensor := harness.UploadVector(source, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer sourceTensor.Close()
				defer outputTensor.Close()

				if err := DispatchDropoutRefs(
					harness.ContextRef(),
					sourceTensor.Ref(),
					outputTensor.Ref(),
					uint32(count),
					config,
					dtype.Float32,
				); err != nil {
					t.Fatalf("dispatch Dropout: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
