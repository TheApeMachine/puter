//go:build cuda

package active_inference

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuactive "github.com/theapemachine/puter/device/cpu/active_inference"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestActiveInferenceCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA belief update", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				likelihood := parity.RandomUnaryInput(count, 0xA100+int64(count))
				prior := parity.RandomUnaryInput(count, 0xA101+int64(count))
				want := make([]float32, count)
				cpuactive.BeliefUpdateF32Generic(&likelihood[0], &prior[0], &want[0], count)
				partialCount := (count + 255) / 256

				likelihoodTensor := harness.UploadVector(likelihood, dtype.Float32)
				priorTensor := harness.UploadVector(prior, dtype.Float32)
				scratchTensor := harness.UploadVector(make([]float32, partialCount), dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer likelihoodTensor.Close()
				defer priorTensor.Close()
				defer scratchTensor.Close()
				defer outputTensor.Close()

				if err := DispatchBeliefUpdate(
					parity.DeviceRef(harness.ContextRef()),
					parity.BufferRef(likelihoodTensor.Ref()),
					parity.BufferRef(priorTensor.Ref()),
					parity.BufferRef(scratchTensor.Ref()),
					parity.BufferRef(outputTensor.Ref()),
					dtype.Float32,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch BeliefUpdate: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}
