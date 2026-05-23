//go:build cuda

package hawkes

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cudadevice "github.com/theapemachine/puter/device/cuda"
	cpuhawkes "github.com/theapemachine/puter/device/cpu/hawkes"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func TestHawkesCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA Hawkes intensity", t, func() {
		mu := float32(0.1)
		alpha := float32(0.5)
		beta := float32(1.0)

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				eventTimes := parity.RandomUnaryInput(count, 0x8100+int64(count))
				queryTimes := parity.RandomUnaryInput(count, 0x8200+int64(count))
				want := make([]float32, count)
				cpuhawkes.HawkesIntensityScalar(eventTimes, queryTimes, want, mu, alpha, beta)

				eventsTensor := harness.UploadVector(eventTimes, dtype.Float32)
				queryTensor := harness.UploadVector(queryTimes, dtype.Float32)
				baselineTensor := harness.UploadVector([]float32{mu}, dtype.Float32)
				alphaTensor := harness.UploadVector([]float32{alpha}, dtype.Float32)
				betaTensor := harness.UploadVector([]float32{beta}, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer eventsTensor.Close()
				defer queryTensor.Close()
				defer baselineTensor.Close()
				defer alphaTensor.Close()
				defer betaTensor.Close()
				defer outputTensor.Close()

				if err := DispatchHawkesIntensity(
					cudadevice.DeviceRef(harness.ContextRef()),
					cudadevice.BufferRef(eventsTensor.Ref()),
					cudadevice.BufferRef(queryTensor.Ref()),
					cudadevice.BufferRef(baselineTensor.Ref()),
					cudadevice.BufferRef(alphaTensor.Ref()),
					cudadevice.BufferRef(betaTensor.Ref()),
					cudadevice.BufferRef(outputTensor.Ref()),
					uint32(count),
					uint32(count),
					dtype.Float32,
				); err != nil {
					t.Fatalf("dispatch HawkesIntensity: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 3)
			})
		}
	})
}
